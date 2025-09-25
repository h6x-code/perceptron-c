#include "data.h"
#include <string.h>

static const float X_xor[8] = {0,0, 0,1, 1,0, 1,1};
static const int   y_xor[4] = {0,1,1,0};

static const float X_and[8] = {0,0, 0,1, 1,0, 1,1};
static const int   y_and[4] = {0,0,0,1};

static const float X_or[8]  = {0,0, 0,1, 1,0, 1,1};
static const int   y_or[4]  = {0,1,1,1};

Dataset load_dataset(const char *name) {
    if (!strcmp(name, "xor")) return (Dataset){X_xor, y_xor, 4, 2, 2};
    if (!strcmp(name, "and")) return (Dataset){X_and, y_and, 4, 2, 2};
    if (!strcmp(name, "or"))  return (Dataset){X_or,  y_or,  4, 2, 2};
    return (Dataset){0,0,0,0,0};
}
