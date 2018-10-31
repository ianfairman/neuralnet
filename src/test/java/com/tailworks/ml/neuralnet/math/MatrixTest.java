package com.tailworks.ml.neuralnet.math;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class MatrixTest {

    /*
    [2 3 4]   [1]    [20]
    [3 4 5] * [2] =  [26]
              [3]
     */
    @Test
    public void testMultiply() {
        Vec v = new Vec(1, 2, 3);
        Matrix W = new Matrix(new double[][]{{2, 3, 4}, {3, 4, 5}});
        Vec result = W.multiply(v);

        assertArrayEquals(new double[]{20, 26}, result.getData(), 0.1);

    }

    @Test
    public void testMap() {
        Matrix W = new Matrix(new double[][]{{2, 3, 4}, {3, 4, 5}});
        W = W.map(value -> 1);

        assertEquals(1, W.getData()[0][0], 0.1);
        assertEquals(1, W.getData()[1][1], 0.1);
    }

    @Test
    public void testScale() {
        Matrix W = new Matrix(new double[][]{{2, 3, 4}, {3, 4, 5}});
        W = W.scale(2);

        assertEquals(6, W.getData()[0][1], 0.1);
        assertEquals(8, W.getData()[1][1], 0.1);
    }

    @Test
    public void testAdd() {
        Matrix U = new Matrix(new double[][]{
                {2, 3},
                {3, 4}
        });
        Matrix V = new Matrix(new double[][]{
                {4, 5},
                {6, 7}
        });
        Matrix R = U.add(V);

        assertEquals(8, R.getData()[0][1], 0.1);
        assertEquals(11, R.getData()[1][1], 0.1);
    }
}