package be.digan.dl.pbil;

import java.util.ArrayList;
import java.util.List;

public class DenseLayer implements Layer {
    List<Node> nodes;

    public DenseLayer(int inputSize, int outputSize) {
        nodes = new ArrayList<>();
        for (int i = 0; i<outputSize; i ++) {
            int start = i * (inputSize + 1);
            nodes.add(new Node(inputSize));
        }
     }

    public double[] calculate(final double[] weights, final double[] input) {
        double[] result = nodes.stream().mapToDouble(n -> n.calculate(weights, input)).toArray();
        return result;
    }

    public int getWeightCount() {
        return nodes.stream().mapToInt(n -> n.getWeightCount()).sum();
    }

    public int setWeightIndex(int index) {
        for (Node node : nodes) {
            int count = node.getWeightCount();
            node.setWeightIndex(index);
            index += count;
        }
        return index;
    }
}
