package com.tailworks.ml.neuralnet.math;


import java.util.Arrays;

import static java.lang.String.format;
import static java.util.Arrays.stream;

/**
 * Careful: not immutable   TODO: consider fixing
 */
public class Matrix {

    private double[][] data;
    private int rows, cols;

    public Matrix(double[][] data) {
        this.data = data;
        rows = data.length;
        cols = data[0].length;
    }

    public Matrix(int rows, int cols) {
        this(new double[rows][cols]);
    }

    public Vec multiply(Vec v) {
        double[] out = new double[rows];
        for (int y = 0; y < rows; y++) {
            out[y] = new Vec(data[y]).dot(v);
        }
        return new Vec(out);
    }

    public Matrix map(Function fn) {
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                data[y][x] = fn.apply(data[y][x]);

        return this;
    }

    public int numberOfRows() {
        return rows;
    }

    public int numberOfCols() {
        return cols;
    }

    public Matrix scale(double s) {
        return map(value -> s * value);
    }

    public double[][] getData() {
        return data;
    }

    public Matrix subtract(Matrix other) {
        assertCorrectDimension(other);

        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                data[y][x] -= other.data[y][x];

        return this;
    }

    public double average() {
        return stream(data).flatMapToDouble(Arrays::stream).average().getAsDouble();
    }

    public double variance() {
        double avg = average();
        return stream(data).flatMapToDouble(Arrays::stream).map(a -> (a - avg) * (a - avg)).average().getAsDouble();
    }


    private void assertCorrectDimension(Matrix other) {
        if (rows != other.rows || cols != other.cols)
            throw new IllegalArgumentException(format("Matrix of different dim: Input is %d x %d, Vec is %d x %d", rows, cols, other.rows, other.cols));
    }
}

