package be.digan.dl.pbil;

import java.util.Arrays;

public class Softmax implements Layer {


    @Override
    public long[] calculate(long[] weights, long[] input) {
        long max = NeuralNet.FACTOR;//Arrays.stream(input).max().getAsLong();

        double[] exp = Arrays.stream(input).mapToDouble(i -> Math.exp((double)i / max)).toArray();
        double sum = Arrays.stream(exp).sum();

        long[] collect = Arrays.stream(exp).mapToLong(i -> (long)((i / sum) * NeuralNet.FACTOR)).toArray();
        return collect;
    }

    @Override
    public int getWeightCount() {
        return 0;
    }


    @Override
    public int setWeightIndex(int weightIndex) {
        return weightIndex;
    }

    @Override
    public int[] appendWeightStructure(int[] baseStructure) {
        return baseStructure;
    }

    @Override
    public String toString() {
        return "Softmax()";
    }

}
