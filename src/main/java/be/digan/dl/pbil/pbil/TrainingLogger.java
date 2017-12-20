package be.digan.dl.pbil.pbil;

public interface TrainingLogger {
    void fullLog(String logId, long[] genotype);
    void fullLog(int generation, long[] genotype);

    void estimate(int generation, double estimatedQuality);
}
