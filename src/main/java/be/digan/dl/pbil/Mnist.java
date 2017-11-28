package be.digan.dl.pbil;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

public class Mnist {
    private static final String MNIST_FOLDER = "/home/david/MNIST";
    private static Experiment[] mnist_test;
    private static Experiment[] mnist_data;

    public static void main(String[] args) throws IOException {
        load_data();
    }

    private static void load_data() throws IOException {
        mnist_test = read_csv(MNIST_FOLDER + "/mnist_test.csv");
        mnist_data = read_csv(MNIST_FOLDER + "/mnist_train.csv");

        PbilNeuralNetTrainer trainer = new PbilNeuralNetTrainer();
        NeuralNet net = getSingleLayerExperiment();

        trainer.trainNetwork(net, mnist_data, mnist_test);

    }

    private static NeuralNet getSingleLayerExperiment() {
        NeuralNet net = new NeuralNet();
        net.addLayer(new DenseLayer(28*28,10));
        net.addLayer(new Softmax());
        return net;
    }
    private static NeuralNet getDualLayerExperiment() {
        NeuralNet net = new NeuralNet();
        net.addLayer(new DenseLayer(28*28,800));
        net.addLayer(new DenseLayer(800,10));
        net.addLayer(new Softmax());
        return net;
    }

    private static Experiment[] read_csv(String file) throws IOException {
        return Files.lines(Paths.get(file)).map(s -> readline(s)).toArray(Experiment[]::new);
    }

    private static Experiment readline(String s) {
        String[] split = s.split(",");
        int output = new Integer(split[0]);
        double[] input = Arrays.stream(split, 1, split.length)
                .map(String::trim).mapToDouble(Integer::parseInt).toArray();
        return new Experiment(input, output);
    }
}
