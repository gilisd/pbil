package be.digan.dl.pbil;

public class Relu implements ActivationFunction {
    @Override
    public long activate(long value) {
        return value<0?0:value;
    }

    @Override
    public String getName() {
        return "RELU";
    }
}
