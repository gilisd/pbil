package be.digan.dl.pbil.pbil;

import be.digan.dl.pbil.neuralnet.NeuralNet;

public interface Validator {
    void newBatch();
    // quality -> higher is better
    double calculateQuality(long[] weights);
}
