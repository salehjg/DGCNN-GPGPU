//
// Created by saleh on 4/23/19.
//

#ifndef DEEPPOINTV1_TIMINGGPU_H
#define DEEPPOINTV1_TIMINGGPU_H

struct PrivateTimingGPU;
class TimingGPU {
private:
    PrivateTimingGPU *privateTimingGPU;

public:

    TimingGPU();

    ~TimingGPU();

    void StartCounter();
    void StartCounterFlags();

    float GetCounter();
};


#endif //DEEPPOINTV1_TIMINGGPU_H
