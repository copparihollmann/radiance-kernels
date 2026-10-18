"""Generate and qualify isolated CPU executions of context-buffer stages."""

import re


def wrap_driver(source, preloads, plan, boundaries=False):
    for old in ('extern unsigned char v_arena[];',
                'int main(int argc, char** argv) {',
                'stages[i].run(nullptr, tid, 32, 0);'):
        if source.count(old) != 1:
            raise ValueError('unrecognized native driver: ' + old)
    source = source.replace('extern unsigned char v_arena[];', '')
    source = source.replace('int main(int argc, char** argv) {',
                            'static int run_replica(int argc, char** argv, unsigned char* v_arena) {')
    source = source.replace('stages[i].run(nullptr, tid, 32, 0);',
                            'stages[i].run(v_arena, tid, 32, 0);')
    for line in preloads:
        if source.count(line) != 1:
            raise ValueError('unrecognized preload in native driver')
        source = source.replace(line, '')
    # Each independent reference run has its own boundary totals.
    if boundaries:
        source = source.replace('  const uint32_t total = sizeof(stages)',
                                '  boundary_bad = boundary_checked = 0;\n'
                                '  const uint32_t total = sizeof(stages)')
    wrapper = r'''
#include <string>
#include <sys/stat.h>
int main(int argc, char** argv) {
PRELOADS
  void* memory = nullptr;
  if (posix_memalign(&memory, 4096, POOL_BYTES)) return 2;
  unsigned char* pool = (unsigned char*)memory;
  memset(pool, 0xa5, POOL_BYTES);
  for (unsigned replica = 0; replica < REPLICAS; ++replica)
    memset(pool + replica * STRIDE + GUARD, 0, BODY);
  const char* trace = getenv("SMOLVLA_NATIVE_TRACE_DIR");
  const std::string trace_root = trace ? trace : "";
  unsigned failures = 0;
  for (unsigned replica = 0; replica < REPLICAS; ++replica) {
    unsigned char* arena = pool + replica * STRIDE + GUARD;
    fprintf(stderr, "REPLICA_BEGIN %u\n", replica);
    if (!trace_root.empty()) {
      std::string path = trace_root + "/replica-" + std::to_string(replica);
      if (mkdir(path.c_str(), 0755) && errno != EEXIST) return 2;
      setenv("SMOLVLA_NATIVE_TRACE_DIR", path.c_str(), 1);
    }
    std::string dump = argc > 2 ? argv[2] : "";
    if (replica && !dump.empty()) dump += ".replica" + std::to_string(replica);
    char* args[] = {argv[0], argc > 1 ? argv[1] : nullptr,
                    dump.empty() ? nullptr : (char*)dump.c_str()};
    int status = run_replica(argc, args, arena);
    fflush(stdout);
    fprintf(stderr, "REPLICA_DONE %u status=%d\n", replica, status);
    fflush(stderr);
    failures |= status != 0;
  }
  unsigned guard_bad = 0;
  for (unsigned replica = 0; replica < REPLICAS; ++replica)
    for (unsigned j = 0; j < GUARD; ++j) {
      guard_bad += pool[replica * STRIDE + j] != 0xa5u;
      guard_bad += pool[replica * STRIDE + GUARD + BODY + j] != 0xa5u;
    }
  printf("REPLICA_GUARDS failed_bytes=%u\n", guard_bad);
  free(memory);
  return failures || guard_bad ? 1 : 0;
}
'''
    # The raw C++ block deliberately uses literal escapes; replace only named
    # whole-word constants, never arbitrary values from paths or checkpoints.
    wrapper = wrapper.replace('PRELOADS', '\n'.join(preloads))
    for key, value in dict(POOL_BYTES=plan['bytes'], REPLICAS=plan['replicas'],
                           STRIDE=plan['stride'], GUARD=plan['guard_bytes'],
                           BODY=plan['body_bytes']).items():
        wrapper = re.sub(r'\b' + key + r'\b', str(value) + 'u', wrapper)
    return '#include <cerrno>\n' + source + wrapper


def validate_log(log, replicas, stages, boundaries=False, original=False, full=True):
    matches = list(re.finditer(r'^REPLICA_BEGIN (\d+)\s*$', log, re.M))
    results = []
    for index, match in enumerate(matches):
        block = log[match.end():matches[index+1].start() if index+1 < len(matches) else len(log)]
        replica = int(match[1])
        done = re.findall(r'^REPLICA_DONE (\d+) status=(\d+)\s*$', block, re.M)
        completed = re.findall(r'completed (\d+)/(\d+):', block)
        actions = re.findall(r'^actions=(\d+) failures=(\d+) first_failure=\d+ max_abs_error=([^\s]+)', block, re.M)
        boundary = re.findall(r'^BOUNDARY_TOTAL checked=(\d+) failed_stages=(\d+)', block, re.M)
        baseline = re.findall(r'^ORIGINAL actions=(\d+) failures=(\d+) max_abs_error=([^\s]+)', block, re.M)
        passed = (done == [(str(replica), '0')] and completed and int(completed[-1][0]) == stages)
        if full:
            passed = passed and len(actions) == 1 and actions[0][0:2] == ('1600', '0')
        if full and boundaries:
            passed = passed and boundary == [('1930', '0')]
        if full and original:
            passed = passed and len(baseline) == 1 and baseline[0][0:2] == ('1600', '0')
        results.append(dict(replica=replica, passed=bool(passed), completed=completed[-1:] or None,
                            format_actions=actions, boundaries=boundary, original_actions=baseline))
    guards = re.findall(r'^REPLICA_GUARDS failed_bytes=(\d+)\s*$', log, re.M)
    passed = (len(results) == replicas and [r['replica'] for r in results] == list(range(replicas))
              and all(r['passed'] for r in results) and guards == ['0'])
    return dict(passed=passed, replicas=results, guard_checks_pass=guards == ['0'],
                scope='complete generated CPU graphs in disjoint guarded buffers; device ISA execution is separate')
