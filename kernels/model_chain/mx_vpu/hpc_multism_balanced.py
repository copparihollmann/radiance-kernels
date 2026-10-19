"""Four-SM launcher with interleaved ownership for small Spatter outputs.

The original generator stays unchanged because earlier immutable execution
receipts pin its exact bytes. This generator is used for the replacement
four-SM workloads; their fixtures and arithmetic helper stay unchanged.
"""

from hpc_multism import kernel_source as original_kernel_source


def kernel_source(family: str) -> str:
    source = original_kernel_source(family, 4)
    if family == 'spatter':
        old = '  spatter_apply<4>(tid, tpb, cluster);'
        assert source.count(old) == 1
        # The helper computes its flat owner index as cluster * tpb + tid
        # and advances by 4 * tpb. Pass an interleaved flat index and zero
        # block offset, retaining that stride and each owner's serial loop.
        source = source.replace(old, '''  // Interleave owners across SMs even when owners < 4 * tpb.
  spatter_apply<4>(tid * 4u + cluster, tpb, 0u);''')
    return source
