package be.digan.dl.pbil.mnist;

import be.digan.dl.pbil.neuralnet.Experiment;
import be.digan.dl.pbil.neuralnet.NeuralNet;
import be.digan.dl.pbil.neuralnet.logger.ByCategoryLogger;
import be.digan.dl.pbil.pbil.TrainingLogger;
import org.apache.log4j.Logger;

import java.util.Arrays;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.joining;

public class MnistLogger implements TrainingLogger {
    private static final Logger LOG = Logger.getLogger(MnistLogger.class);
    private final Experiment[] data;
    private final NeuralNet net;
    private ByCategoryLogger byCategoryLogger;

    public MnistLogger(Experiment[] experiments, NeuralNet net) {
        this.data = experiments;
        this.net = net;
        byCategoryLogger = new ByCategoryLogger(experiments);
    }

    @Override
    public void fullLog(int generation, long[] genotype) {
        String generationString = "Generation " + String.format("%5d", generation);
        fullLog(generationString, genotype);

    }

    @Override
    public void estimate(int generation, double estimatedQuality) {
        LOG.debug("Generation " + String.format("%5d", generation) + ": estimated: " + String.format("%10f", estimatedQuality));
    }

    @Override
    public void fullLog(String generationString, long[] genotype) {
        byCategoryLogger.clear();
        double totalQuality = 0;
        int[] histo = new int[11];
        for (int j = 0; j < data.length; j++) {
            int element = j;
            long[] calculate = net.calculate(genotype, data[element].getInput());
            double result = (double) calculate[data[element].getOutput()] / NeuralNet.FACTOR;
            double quality = result;

            int found = IntStream.range(0, calculate.length)
                    .reduce((a, b) -> calculate[a] < calculate[b] ? b : a)
                    .getAsInt();
            histo[(int) (result * 10)] = histo[(int) (result * 10)] + 1;
            byCategoryLogger.append(found, data[element].getOutput(), calculate);
            // quality: closer to one is better
            totalQuality += quality;
        }
        double quality = totalQuality / data.length;
        double averageAbs = Arrays.stream(genotype).map(i -> Math.abs(i)).average().getAsDouble();
        double average = Arrays.stream(genotype).average().getAsDouble();
        long min = Arrays.stream(genotype).min().getAsLong();
        long max = Arrays.stream(genotype).max().getAsLong();
        LOG.info(generationString + ": average abs: " + String.format("%10f", averageAbs) + ": average: " + String.format("%10f", average) + ", max: " + max + ", min: " + min);
        LOG.info(generationString + ": random: " + data[500].getOutput() + " " + writeArray(net.calculate(genotype, data[500].getInput())));
        LOG.info(generationString + ": deviation: " + String.format("%10f", quality) + " histogram: " + writeArray(histo) + ", "+byCategoryLogger.out());
    }

    private String writeArray(int[] histo) {
        long sum = Arrays.stream(histo).sum();
        return "[" + Arrays.stream(histo).mapToObj(i -> String.format("%3d%%", (i * 100 / sum))).collect(joining(", ")) + "]";
    }


    private String writeArray(long[] histo) {
        long sum = Arrays.stream(histo).sum();
        return "[" + Arrays.stream(histo).mapToObj(i -> sum==0?"NAN": String.format("%3d%%", (i * 100 / sum))).collect(joining(", ")) + "]";
    }

    private String writeArray(int[] histo, int digits) {
        long sum = Arrays.stream(histo).sum();
        return "[" + Arrays.stream(histo).mapToObj(i -> String.format("%" + digits + "d", i)).collect(joining(", ")) + "]";
    }

    private String writeArray(long[] histo, int digits) {
        long sum = Arrays.stream(histo).sum();
        return "[" + Arrays.stream(histo).mapToObj(i -> String.format("%" + digits + "d", i)).collect(joining(", ")) + "]";
    }


}
