package be.digan.dl.pbil.neuralnet;

import be.digan.dl.pbil.neuralnet.layer.Layer;
import org.apache.log4j.Logger;

import java.util.List;
import java.util.ArrayList;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.joining;

public class NeuralNet {
    private static final Logger LOG = Logger.getLogger(NeuralNet.class);
    public static final int DIGITS = 4;
    public static final int FACTOR = (int)Math.pow(10, DIGITS);

    int currentWeightIndex = 0;
    private List<Layer> layers = new ArrayList();
    private int weightCount;
    private int[] weightStructure;

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
        calculateStructure();
    }

    private void calculateStructure() {
        weightCount = calculateWeightCount();
        weightStructure = calculateWeightStructure();
    }

    public int calculateWeightCount() {
        return layers.stream().mapToInt(l -> l.getWeightCount()).sum();
    }

    @Override
    public String toString() {
        return layers.stream().map(l-> l.toString()).collect(joining(" - "));
    }

    public int[] calculateWeightStructure() {
        int[] structure = new int[0];
        for (Layer layer : layers) {
            structure = layer.appendWeightStructure(structure);
        }
        return structure;
    }

    public int getWeightCount() {
        return weightCount;
    }

    public int[] getWeightStructure() {
        return weightStructure;
    }

    public void log() {
        LOG.info("NETWORK STRUCTURE: ");
        LOG.info(" - chromosoneSize: " + getWeightCount());
        LOG.info(" - layers: " + toString());
    }

    public static int getMaxIndex(int[] calculate) {
        return IntStream.range(0, calculate.length)
                .reduce((a,b)->calculate[a]<calculate[b]? b: a)
                .getAsInt();
    }
}
