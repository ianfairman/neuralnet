package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Matrix;
import com.tailworks.ml.neuralnet.math.Vec;

import static java.lang.String.format;


public class Layer {

    private final int size;
    private ThreadLocal<Vec> out = new ThreadLocal<>();
    private Activation activation;
    private Matrix weights;
    private Vec bias;

    private transient Matrix deltaWeights;
    private transient Vec deltaBias;
    private transient int deltaWeightsAdded = 0;
    private transient int deltaBiasAdded = 0;

    private Layer precedingLayer;


    public Layer(int size, Activation activation) {
        this(size, activation, 0);
    }

    public Layer(int size, Activation activation, double initialBias) {
        this.size = size;
        bias = new Vec(size).map(x -> initialBias);
        deltaBias = new Vec(size);
        this.activation = activation;
    }

    public Layer(int size, Activation activation, Vec bias) {
        this.size = size;
        this.bias = bias.copy();
        deltaBias = new Vec(size);
        this.activation = activation;
    }

    public int size() {
        return size;
    }

    public Layer evaluate(Vec in) {
        if (weights != null) {
            in = in.mul(weights);
        }

        if (in.dimension() != size)
            throw new IllegalArgumentException(format("Input data is of size %d while network expects vectors of size %d", in.dimension(), size));

        out.set(activation.fn(in.add(bias)));

        return this;
    }

    public Vec getOut() {
        return out.get();
    }

    public Activation getActivation() {
        return activation;
    }

    public void addWeights(Matrix weights) {
        this.weights = weights;
        deltaWeights = new Matrix(weights.rows(), weights.cols());
    }

    public Matrix getWeights() {
        return weights;
    }

    public Layer getPrecedingLayer() {
        return precedingLayer;
    }

    public void setPrecedingLayer(Layer precedingLayer) {
        this.precedingLayer = precedingLayer;
    }

    public boolean hasPrecedingLayer() {
        return precedingLayer != null;
    }

    public boolean isInputLayer() {
        return precedingLayer == null;
    }

    public Vec getBias() {
        return bias;
    }

    public synchronized void addDeltaWeights(Matrix dW) {
        deltaWeights.add(dW);
        deltaWeightsAdded++;
    }

    public synchronized void addDeltaBias(Vec dB) {
        deltaBias = deltaBias.add(dB);
        deltaBiasAdded++;
    }

    public synchronized void updateWeights(double learningRate) {
        if (deltaWeightsAdded == 0) return;

        Matrix average_dW = deltaWeights.mul(1.0 / deltaWeightsAdded);
        weights.sub(average_dW.mul(learningRate));

        deltaWeights.map(a -> 0);
        deltaWeightsAdded = 0;
    }

    public synchronized void updateBias(double learningRate) {
        if (deltaBiasAdded == 0) return;

        Vec average_bias = deltaBias.mul(1.0 / deltaBiasAdded);
        bias = bias.sub(average_bias.mul(learningRate));

        deltaBias = deltaBias.map(a -> 0);
        deltaBiasAdded = 0;
    }


    // ------------------------------------------------------------------


    public LayerState getState() {
        return new LayerState(this);
    }

    public static class LayerState {

        double[][] weights;
        double[] bias;
        String activation;

        public LayerState(Layer layer) {
            weights = layer.getWeights() != null ? layer.getWeights().getData() : null;
            bias = layer.getBias().getData();
            activation = layer.activation.getName();
        }

        public double[][] getWeights() {
            return weights;
        }

        public double[] getBias() {
            return bias;
        }
    }
}
