#!/usr/bin/env python3
"""Generate a four-SM startup that waits for a late host tag before stack use.

The separate source preserves runtime/input hashes of already running models.
This only applies to a runtime built with MU_CLUSTER_ID_FROM_PRINTBUF=1.
"""
import argparse
from pathlib import Path


def require_tag(source):
    old='''  li    t1, 0x80038
  lw.shared t3, 0(t1)
  srli  t2, t3, 16
  li    t1, 0xc1d0
  bne   t2, t1, .Lcluster_hart_ready
'''
    new='''  # The GPU may boot before the host. No stack or TLS may use the
  # aliased CSR identity while the physical cluster tag is still missing.
.Lwait_cluster_tag:
  li    t1, 0x80038
  lw.shared t3, 0(t1)
  srli  t2, t3, 16
  li    t1, 0xc1d0
  bne   t2, t1, .Lwait_cluster_tag
'''
    if source.count(old)!=1:
        raise ValueError('startup changed; review the tag-wait transformation')
    return source.replace(old,new)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.write_text(require_tag(args.source.read_text()))
