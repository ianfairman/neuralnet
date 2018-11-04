package com.tailworks.ml.neuralnet;

import com.google.gson.Gson;
import com.tailworks.ml.neuralnet.math.Matrix;
import com.tailworks.ml.neuralnet.math.Vec;
import com.tailworks.ml.neuralnet.util.Util;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private final double learningRate;
    private List<Layer> layers = new ArrayList<>();
    private List<Matrix> deltaWeights = new ArrayList<>();
    private List<Vec> deltaBias = new ArrayList<>();
    private CostFunction costFunction;


    private NeuralNetwork(Builder nb) {
        learningRate = nb.learningRate;
        costFunction = nb.costFunction;

        // Adding inputLayer
        Layer inputLayer = new Layer(nb.networkInputSize, Activation.Identity);
        layers.add(inputLayer);

        // int sizeOfPreviousLayer = networkInputSize;
        Layer precedingLayer = inputLayer;

        for (int i = 0; i < nb.layers.size(); i++) {
            Layer layer = nb.layers.get(i);
            Matrix w = new Matrix(layer.size(), precedingLayer.size());
            nb.initializer.initWeights(w, i);
            layer.addWeights(w);
            layers.add(layer);
            layer.setPrecedingLayer(precedingLayer);
            precedingLayer = layer;
        }
    }


    public Result evaluate(Vec input) {
        return evaluate(input, null);
    }


    public Result evaluate(Vec input, Vec wanted) {
        Vec signal = input;
        for (Layer layer : layers)
            signal = layer.feedWith(signal).getOut();


        if (wanted != null) {
            double cost = costFunction.getTotal(wanted, signal);
            return new Result(signal, cost);
        }
        return new Result(signal);
    }


    public void learn(Vec wanted) {
        Layer lastLayer = getLastLayer();
        Layer layer = lastLayer;

        do {
            Layer precedingLayer = layer.getPrecedingLayer();
            Vec out = layer.getOut();

            Vec dE_dO = layer == lastLayer ?        // Change of Error w.r.p change of output
                    costFunction.getDerivative(wanted, out) :
                    layer.getBackpropError();  // error prepared in last layer for us

            Vec dO_dI = layer.getActivation().dFn(out);

            // prepare error propagation and store in
            Vec dE_dI = dE_dO.elementProduct(dO_dI);
            Vec backpropErrorToNextLayer = layer.getWeights().transpose().multiply(dE_dI);
            precedingLayer.setBackpropError(backpropErrorToNextLayer);

            Matrix dE_dW = precedingLayer.getOut().outerProduct(dE_dI);

            deltaWeights.add(0, dE_dW);
            deltaBias.add(0, dE_dI);

            layer = precedingLayer;
        }
        while (layer.hasPrecedingLayer());      // Stop when we are at input layer


        // ----------------------------------
        // Update weights
        // ----------------------------------
        int cnt = 0;
        for (Layer l : layers)
            if (notFirstLayer(l)) {
                l.getWeights().subtract(deltaWeights.get(cnt).scale(learningRate));
                if (l.hasBias()) {
                    Vec newBias = l.getBias().subtract(deltaBias.get(cnt).scale(learningRate));
                    l.setBias(newBias);
                }
                cnt++;
            }

        deltaWeights.clear();
        deltaBias.clear();
    }

    private boolean notFirstLayer(Layer layer) {
        return layer.getPrecedingLayer() != null;
    }


    private Layer getLastLayer() {
        return layers.get(layers.size() - 1);
    }

    public void printWeights() {
        int cnt = 1;
        for (Layer layer : layers) {
            if (notFirstLayer(layer)) {
                Matrix weights = layer.getWeights();
                double[][] data = weights.transpose().getData();
                System.out.println("W" + (cnt++) + ": \n" + Util.prettyString(data));
            }
        }
    }


    public String toJson() {
        return new Gson().toJson(new NetworkState(this));
    }

    // --------------------------------------------------------------------

    /**
     * Simple builder for a NeuralNetwork
     */
    public static class Builder {

        private List<Layer> layers = new ArrayList<>();
        private int networkInputSize;

        // defaults:
        private double learningRate = 0.25;
        private Initializer initializer = new Initializer.Random(0.5, 2);
        private CostFunction costFunction = new CostFunction.Quadratic();

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
            return new NeuralNetwork(this);
        }

    }

    // -----------------------------

    private static class NetworkState {
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
    }
}

