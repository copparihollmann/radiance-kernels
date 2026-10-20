#ifndef __MU_SCHEDULE_H__
#define __MU_SCHEDULE_H__

#include <stdint.h>

extern "C" {

typedef void (*mu_schedule_callback)(void *arg,
                                     uint32_t tid_in_threadblock,
                                     uint32_t threads_per_threadblock,
                                     uint32_t threadblock_id);

/* Runtime layers may override the default CSR-based identity. */
uint32_t mu_cluster_id(void);

void mu_schedule(mu_schedule_callback entry_point,
                 void * arg,
                 const uint32_t occupancy);

} // extern "C"

#endif // __MU_SCHEDULE_H__
