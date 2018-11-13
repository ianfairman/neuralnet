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
        W = W.mul(2);

        assertEquals(6, W.getData()[0][1], 0.1);
        assertEquals(8, W.getData()[1][1], 0.1);
    }

    @Test
    public void testAverage() {
        Matrix U = new Matrix(new double[][]{
                {2, 3},
                {3, 4}
        });
        Matrix V = new Matrix(new double[][]{
                {4, 5},
                {6, 7}
        });

        assertEquals(3, U.average(), 0.1);
        assertEquals(5.5, V.average(), 0.1);
    }

    @Test
    public void testVariance() {
        Matrix U = new Matrix(new double[][]{
                {11, 3},
                {3, 7}
        });
        //noinspection IntegerDivisionInFloatingPointContext,PointlessArithmeticExpression
        assertEquals(((11-6)*(11-6) + (6-3)*(6-3) + (6-3)*(6-3) + (7-6)*(7-6)) / 4, U.variance(), 0.1);
    }

}