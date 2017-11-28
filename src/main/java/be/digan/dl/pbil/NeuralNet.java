package be.digan.dl.pbil;

import java.util.ArrayList;
import java.util.List;

public class NeuralNet {
    int currentWeightIndex = 0;
    private List<Layer> layers = new ArrayList();
    public double[] calculate(double[] weights, double[] input) {
        double[] intermediate = input;
        for (Layer layer : layers) {
            intermediate = layer.calculate(weights, intermediate);
        }
        return intermediate;
    }

    public void addLayer(Layer layer) {
        currentWeightIndex += layer.setWeightIndex(currentWeightIndex);
        this.layers.add(layer);
    }

    public int getWeightCount() {
        return layers.stream().mapToInt(l -> l.getWeightCount()).sum();
    }
}
