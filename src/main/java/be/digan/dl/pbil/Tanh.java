package be.digan.dl.pbil;

public class Tanh implements ActivationFunction {
    @Override
    public long activate(long value) {
        return (long)(Math.tanh((double)value / NeuralNet.FACTOR) * NeuralNet.FACTOR);
    }

    @Override
    public String getName() {
        return "TANH";
    }
}
