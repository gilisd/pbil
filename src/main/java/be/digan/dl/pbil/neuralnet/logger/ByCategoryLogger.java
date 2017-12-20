package be.digan.dl.pbil.neuralnet.logger;

import be.digan.dl.pbil.neuralnet.Experiment;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.joining;

public class ByCategoryLogger {
    private final int[] compare;
    private int[] byDigit;

    public ByCategoryLogger(Experiment[] data) {
        Map<Integer, List<Integer>> temp = IntStream.range(0, data.length).mapToObj(i -> i).collect(Collectors.groupingBy((Integer i) -> data[i].getOutput()));
        compare = temp.entrySet().stream().mapToInt(i -> i.getValue().size()).toArray();
    }

    public void append(int found, int expected, long[] calculate) {
        byDigit[expected] += found == expected ? 1 : 0;
    }

    public void clear() {
        byDigit = new int[compare.length];
    }

    public String out() {
        return "by digit " + writeArray() + " - " + overall();
    }

    private String overall() {
        int total = Arrays.stream(byDigit).sum();
        int max = Arrays.stream(compare).sum();
        return String.format("%3d%%", 100 * total / max);
    }

    private String writeArray() {
        return "[" + IntStream.range(0, compare.length).mapToObj(i -> String.format("%3d%%", (byDigit[i] * 100 / compare[i]))).collect(joining(", ")) + "]";
    }

}
