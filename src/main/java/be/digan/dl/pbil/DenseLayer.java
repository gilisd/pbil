package be.digan.dl.pbil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class DenseLayer implements Layer {
    private final int weightCount;
    List<Node> nodes;

    public DenseLayer(int inputSize, int outputSize, ActivationFunction activation) {
        nodes = new ArrayList<>();
        for (int i = 0; i<outputSize; i ++) {
            int start = i * (inputSize + 1);
            nodes.add(new Node(inputSize, activation));
        }
        weightCount = nodes.stream().mapToInt(n -> n.getWeightCount()).sum();
     }

    public long[] calculate(final long[] weights, final long[] input) {
        long[] result = nodes.stream().mapToLong(n -> n.calculate(weights, input)).toArray();
        return result;
    }

    public int getWeightCount() {
        return weightCount;
    }

    public int setWeightIndex(int index) {
        for (Node node : nodes) {
            int count = node.getWeightCount();
            node.setWeightIndex(index);
            index += count;
        }
        return index;
    }

    @Override
    public int[] appendWeightStructure(int[] baseStructure) {
        int nextInt = baseStructure.length>0?baseStructure[baseStructure.length-1]:0;
        IntStream list = Arrays.stream(baseStructure);
        for (Node node : nodes) {
            int finalNextInt = nextInt;
            list = IntStream.concat(list, IntStream.range(0, node.getWeightCount()).map(i-> finalNextInt));
            nextInt++;
        }

        return list.toArray();
    }

    @Override
    public String toString() {
        return "DenseLayer(in: "+ (nodes.get(0).getWeightCount() -1) + ", out: " + nodes.size() + ", " + nodes.get(0).getActivation().getName() + ")";
    }
}
