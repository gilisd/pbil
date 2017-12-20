package be.digan.dl.pbil.neuralnet.activation;

public interface ActivationFunction {
    long activate(long value);
    String getName();
}
