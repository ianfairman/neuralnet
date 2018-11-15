package com.tailworks.ml.neuralnet;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializer;
import com.tailworks.ml.neuralnet.math.Matrix;
import com.tailworks.ml.neuralnet.math.Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private final double learningRate;
    private final CostFunction costFunction;
    private List<Layer> layers = new ArrayList<>();

    /**
     * Creates a neural network given the configuration set in the builder
     * @param nb The config for the neural network
     */
    private NeuralNetwork(Builder nb) {
        learningRate = nb.learningRate;
        costFunction = nb.costFunction;

        // Adding inputLayer
        Layer inputLayer = new Layer(nb.networkInputSize, Activation.Identity);
        layers.add(inputLayer);

        Layer precedingLayer = inputLayer;

        for (int i = 0; i < nb.layers.size(); i++) {
            Layer layer = nb.layers.get(i);
            Matrix w = new Matrix(precedingLayer.size(), layer.size());
            nb.initializer.initWeights(w, i);
            layer.addWeights(w);    // Each layer contains the weights between preceding and itself
            layer.setPrecedingLayer(precedingLayer);
            layers.add(layer);

            precedingLayer = layer;
        }
    }


    /**
     * Evaluates an input vector, returning the networks output,
     * without cost or learning anything from it.
     */
    public Result evaluate(Vec input) {
        return evaluate(input, null);
    }


    /**
     * Evaluates an input vector, returning the networks output.
     * If <code>wanted</code> is specified the result will contain
     * a cost and the network will gather some learning from this
     * operation.
     */
    public Result evaluate(Vec input, Vec wanted) {
        Vec signal = input;
        for (Layer layer : layers)
            signal = layer.evaluate(signal).getOut();

        if (wanted != null) {
            learnFrom(wanted);
            double cost = costFunction.getTotal(wanted, signal);
            return new Result(signal, cost);
        }

        return new Result(signal);
    }


    /**
     * Will gather some learning based on the <code>wanted</code> vector
     * and how that differs to the actual output from the network. This
     * difference (or error) is backpropagated through the net. To make
     * it possible to use mini batches the learning is not immediately
     * realized - i.e. <code>learnFrom</code> does not alter any weights.
     * Use <code>updateFromLearning()</code> to do that.
     */
    private void learnFrom(Vec wanted) {
        // We'll start at the last layer
        Layer layer = getLastLayer();

        // The error is initially the derivative of cost-function.
        Vec dEdO = costFunction.getDerivative(wanted, layer.getOut());

        // iterate backwards through the layers
        do {
            Vec dEdI = layer.getActivation().dEdI(layer.getOut(), dEdO);

            // prepare error propagation and store for next iteration
            dEdO = layer.getWeights().multiply(dEdI);

            Matrix dEdW = dEdI.outerProduct(layer.getPrecedingLayer().getOut());

            // Store the deltas per layer (i.e. the weight changes we want)
            layer.addDeltaWeights(dEdW);
            layer.addDeltaBias(dEdI);

            layer = layer.getPrecedingLayer();
        }
        while (layer.hasPrecedingLayer());      // Stop when we are at input layer
    }


    /**
     * Let all gathered (but not yet realised) learning "sink in".
     * That is: Update the weights and biases based on the deltas
     * collected during evaluation & training.
     */
    public void updateFromLearning() {
        for (Layer l : layers) {
            if (!l.isInputLayer()) {
                l.updateWeights(learningRate);
                l.updateBias(learningRate);
            }
        }
    }



    public List<Layer> getLayers() {
        return layers;
    }

    public String toJson(boolean pretty) {
        GsonBuilder gsonBuilder = new GsonBuilder()
                .registerTypeAdapter(Double.class,
                        (JsonSerializer<Double>) (src, typeOfSrc, context) ->
                                new JsonPrimitive(src.floatValue())
                );
        if (pretty) gsonBuilder.setPrettyPrinting();
        return gsonBuilder.create().toJson(new NetworkState(this));
    }


    private Layer getLastLayer() {
        return layers.get(layers.size() - 1);
    }


    // --------------------------------------------------------------------

    /**
     * Simple builder for a NeuralNetwork
     */
    public static class Builder {
        private static Logger log = LoggerFactory.getLogger(Builder.class);

        private List<Layer> layers = new ArrayList<>();
        private int networkInputSize;

        // defaults:
        private double learningRate = 0.005;
        private Initializer initializer = new Initializer.Random(-0.5, 0.5);
        private CostFunction costFunction = new CostFunction.MSE();

        public Builder(int networkInputSize) {
            this.networkInputSize = networkInputSize;
        }

        public Builder setLearningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder initWeights(Initializer initializer) {
            this.initializer = initializer;
            return this;
        }

        public Builder setCostFunction(CostFunction costFunction) {
            this.costFunction = costFunction;
            return this;
        }

        public Builder addLayer(Layer layer) {
            layers.add(layer);
            return this;
        }

        public NeuralNetwork create() {
            NeuralNetwork network = new NeuralNetwork(this);
            log.info("Created NeuralNetwork: " + network.toJson(false).substring(0, 200) + " ...");
            return network;
        }

    }

    // -----------------------------

    public static class NetworkState {
        double learningRate;
        String costFunction;
        Layer.LayerState[] layers;

        public NetworkState(NeuralNetwork network) {
            learningRate = network.learningRate;
            costFunction = network.costFunction.getName();

            layers = new Layer.LayerState[network.layers.size()];
            for (int l = 0; l < network.layers.size(); l++) {
                layers[l] = network.layers.get(l).getState();
            }
        }

        public Layer.LayerState[] getLayers() {
            return layers;
        }
    }
}

