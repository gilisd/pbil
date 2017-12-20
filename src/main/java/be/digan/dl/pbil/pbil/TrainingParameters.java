package be.digan.dl.pbil.pbil;

import org.apache.log4j.Logger;

public class TrainingParameters {
    private static final Logger LOG = Logger.getLogger(TrainingParameters.class);
    private static final int GENERATION_COUNT = 1000;
    private static final int POPULATION = 500;
    private static final int BATCH_SIZE = 150;
    private static final boolean REDUCE_AVERAGE_WEIGHT = true;
    public static final double DEVIATION = .1;

    public static int getGenerationCount() {
        return GENERATION_COUNT;
    }

    public static int getBatchSize() {
        return BATCH_SIZE;
    }

    public static int getPopulation() {
        return POPULATION;
    }

    public void log() {
        LOG.info("TRAINING PARAMETERS: ");
        LOG.info("  - generations: " + getGenerationCount());
        LOG.info("  - population: " + getPopulation());
        LOG.info("  - batch size: " + getBatchSize());
        LOG.info("  - deviation: " + DEVIATION);
        LOG.info("  - reduce average weight: " + reduceAverageWeight());
    }

    public boolean reduceAverageWeight() {
        return REDUCE_AVERAGE_WEIGHT;
    }

    public double getDeviation(int generation) {
        return DEVIATION * (1 - ((double)generation / GENERATION_COUNT));
    }
}
