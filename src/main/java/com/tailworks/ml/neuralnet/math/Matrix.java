package com.tailworks.ml.neuralnet.math;


import static java.lang.String.format;

/**
 * Careful: not immutable
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

    public Matrix scale(double s) {
        return map(value -> s * value);
    }

    public Matrix transpose() {
        double[][] result = new double[cols][rows];
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                result[x][y] = data[y][x];

        return new Matrix(result);
    }

    public int numberOfRows() {
        return rows;
    }

    public int numberOfCols() {
        return cols;
    }

    public double[][] getData() {
        return data;
    }

    public Matrix add(Matrix other) {
        if (rows != other.rows || cols != other.cols)
            throw new IllegalArgumentException(format("Matrix of different dim: Input is %d x %d, Vec is %d x %d", rows, cols, other.rows, other.cols));

        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                data[y][x] += other.data[y][x];

        return this;
    }

    public Matrix subtract(Matrix other) {
        if (rows != other.rows || cols != other.cols)
            throw new IllegalArgumentException(format("Matrix of different dim: Input is %d x %d, Vec is %d x %d", rows, cols, other.rows, other.cols));

        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                data[y][x] -= other.data[y][x];

        return this;
    }

}

