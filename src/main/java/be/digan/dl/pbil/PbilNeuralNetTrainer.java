package be.digan.dl.pbil;

import be.digan.dl.pbil.neuralnet.NeuralNet;
import be.digan.dl.pbil.pbil.TrainingLogger;
import be.digan.dl.pbil.pbil.TrainingParameters;
import be.digan.dl.pbil.pbil.Validator;
import org.apache.log4j.Logger;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.joining;

// Reduce memory load by evaluating item by item
public class PbilNeuralNetTrainer {
    private static final Logger LOG = Logger.getLogger(PbilNeuralNetTrainer.class);
    private TrainingParameters parameters;

    private Random random = new Random();
    private NeuralNet net;
    private Validator validator;
    private TrainingLogger logger;

    public PbilNeuralNetTrainer(TrainingParameters parameters, NeuralNet net, Validator validator, TrainingLogger logger) {
        this.parameters = parameters;
        this.net = net;
        this.validator = validator;
        this.logger = logger;
        parameters.log();
        net.log();
    }

    public long[] trainNetwork() {

        // Pick initial genotype by random search
        long[] genotype = getInitialGenotype();
        logger.fullLog(0, genotype);

        for (int generation = 1; generation <= parameters.getGenerationCount(); generation++) {
            genotype = nextGeneration(genotype, generation);
            if ((generation < 10) || ((generation < 1000) && (generation % 10 == 0)) || (generation % 100 == 0)) {
                logger.fullLog(generation, genotype);
            }
        }

        return genotype;
    }

    private long[] nextGeneration(long[] genotype, int generation) {
        if (parameters.reduceAverageWeight()) {
        if (Arrays.stream(genotype).map(i -> Math.abs(i)).average().getAsDouble() > NeuralNet.FACTOR *parameters.getDeviation(0) *100) { //NORMALIZE
            genotype = Arrays.stream(genotype).map(y -> y / 2).toArray();
            LOG.warn("DIVISION " + generation);
        }
        }
        validator.newBatch();
        long[] currentBestWeights = genotype;
        double currentBestQuality = validator.calculateQuality(currentBestWeights);
        for (int j = 0; j < parameters.getPopulation(); j++) {

            long[] newWeights = generatePbil(generation, genotype);
            double newQuality = validator.calculateQuality(newWeights);
            double difference = newQuality - currentBestQuality;
            boolean isBetter = difference > 0;
            if (isBetter) {
                currentBestWeights = newWeights;
                currentBestQuality = newQuality;
            }
        }
        genotype = currentBestWeights;
        logger.estimate(generation, currentBestQuality);
        return genotype;
    }

    private long[] getInitialGenotype() {
        long[] bestWeights = getRandom();
        validator.newBatch();
        double best = validator.calculateQuality(bestWeights);
        for (int i = 0; i < parameters.getPopulation(); i++) {
            long[] newWeights = getRandom();
            double newQuality = validator.calculateQuality(newWeights);
            double difference = newQuality - best;
            boolean isBetter = difference > 0;
            if (isBetter) {
                bestWeights = newWeights;
                best = newQuality;
            }
        }
        return bestWeights;
    }


    private long[] getRandom() {
        return IntStream.range(0, net.getWeightCount()).mapToLong(k -> (getRandomDeviation(0))).toArray();
    }


    private long[] generatePbilOLD(int generation, long[] best) {
        int[] weightStructure = net.getWeightStructure();
        int geneToChange = random.nextInt(weightStructure[weightStructure.length - 1] + 1);
        long[] result = IntStream.range(0, best.length).mapToLong(i -> map(generation, i, best, weightStructure, geneToChange)).toArray();
        return result;
    }

    private long map(int generation, int i, long[] best, int[] weightStructure, int geneToChange) {
        return best[i] + (weightStructure[i] != geneToChange ? 0 : getRandomDeviation(generation));
    }

    private long[] generatePbil(int generation, long[] best) {
        int[] weightStructure = net.getWeightStructure();
        Boolean[] keepGene = IntStream.range(0, weightStructure[weightStructure.length - 1] + 1).mapToObj(i -> (Boolean) (Math.random() > .8)).toArray(Boolean[]::new);
        long[] result = IntStream.range(0, best.length).mapToLong(i -> best[i] + (keepGene[weightStructure[i]] ? 0 : getRandomDeviation(generation))).toArray();
        return result;
    }

    private long getRandomDeviation(int generation) {
        return random.nextInt((int)(NeuralNet.FACTOR * 2 * parameters.getDeviation(0) +1)) - (long)(NeuralNet.FACTOR * parameters.getDeviation(0));
    }

}
