//
// Created by saleh on 8/22/18.
//

#ifndef DEEPPOINTV1_CUDAIMPLEMENTATION_H
#define DEEPPOINTV1_CUDAIMPLEMENTATION_H

#include <PlatformImplementation.h>

class CudaImplementation: public PlatformImplementation {
public:
    CudaImplementation(int aa); ///TODO: Constructor should handle platform initialization procedure!

private:
    int a;
};


#endif //DEEPPOINTV1_CUDAIMPLEMENTATION_H
