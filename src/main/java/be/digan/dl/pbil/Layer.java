package be.digan.dl.pbil;

public interface Layer {
    long[] calculate(final long[] weights, final long[] input);

    int getWeightCount();

    int setWeightIndex(int index);

    int[] appendWeightStructure(int[] baseStructure);
}
