package be.digan.dl.pbil;

import java.util.ArrayList;
import java.util.List;

import static java.util.stream.Collectors.joining;

public class NeuralNet {
    public static final int DIGITS = 6;
    public static final int FACTOR = (int)Math.pow(10, DIGITS);

    int currentWeightIndex = 0;
    private List<Layer> layers = new ArrayList();
    public long[] calculate(long[] weights, long[] input) {
        long[] intermediate = input;
        for (Layer layer : layers) {
            intermediate = layer.calculate(weights, intermediate);
        }
        return intermediate;
    }

    public void addLayer(Layer layer) {
        currentWeightIndex = layer.setWeightIndex(currentWeightIndex);
        this.layers.add(layer);
    }

    public int getWeightCount() {
        return layers.stream().mapToInt(l -> l.getWeightCount()).sum();
    }

    @Override
    public String toString() {
        return layers.stream().map(l-> l.toString()).collect(joining(" - "));
    }

    public int[] getWeightStructure() {
        int[] structure = new int[0];
        for (Layer layer : layers) {
            structure = layer.appendWeightStructure(structure);
        }
        return structure;
    }
}
