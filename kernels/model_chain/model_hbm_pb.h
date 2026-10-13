/* Extra printBuf slots used only by the model-chain HBM handoff. */
#ifndef MODEL_HBM_PB_H
#define MODEL_HBM_PB_H

#include <nightly/pb.h>

/* The standard START/END slots remain the low 32 bits and END remains the
 * completion marker. Slots 25 and 26 are outside nightly verification slots
 * 16..24 and the model heartbeat slot 48. The generated model uses core 0. */
#define MODEL_HBM_PB_START_HIGH 25u
#define MODEL_HBM_PB_END_HIGH   26u

#endif
