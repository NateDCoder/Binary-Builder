function dec2Bin(dec) {
    if (dec >= 256) {
        dec = 255; // Cap at 255 if the number exceeds 8-bit limit
    }
    if (dec >= 0) {
        return dec.toString(2).padStart(8, "0"); // Add leading zeros for 8-bit binary
    } else {
        let binary = ((~-dec + 1) >>> 0).toString(2); // Handle negative numbers with 2's complement
        return binary.slice(-8).padStart(8, "0"); // Ensure it's 8 bits
    }
}

function bin2Dec(bin) {
    if (bin.length !== 8) {
        console.error("Binary string must be 8 bits long.");
        return null;
    }

    // Handle negative 2's complement conversion

    return parseInt(bin, 2);
}

function isPointInRect(px, py, rx, ry, rw, rh) {
    return px >= rx && px <= rx + rw && py >= ry && py <= ry + rh;
}

function replaceBit(bin, position, newBit) {
    if (position < 0 || position >= bin.length) {
        console.error("Invalid position. Must be between 0 and " + (bin.length - 1));
        return bin;
    }
    if (newBit !== "0" && newBit !== "1") {
        console.error("Invalid bit. Must be '0' or '1'.");
        return bin;
    }

    // Replace the bit at the specified position and return the modified string
    return bin.substring(0, position) + newBit + bin.substring(position + 1);
}

function seededRandom(seed) {
    let state = seed % 2147483647;
    if (state <= 0) state += 2147483646;

    return function () {
        state = (state * 16807) % 2147483647;
        return (state - 1) / 2147483646;
    };
}

const rng = seededRandom(12345); // Seed value

function softmax(matrix, dim = 1) {
    const exp = (x) => Math.exp(x);

    if (dim === 0) {
        // Column-wise softmax
        const numRows = matrix.length;
        const numCols = matrix[0].length;

        // Transpose, apply softmax row-wise, then transpose back
        let transposed = Array.from({ length: numCols }, (_, i) => matrix.map((row) => row[i]));

        transposed = transposed.map((col) => {
            const maxVal = Math.max(...col);
            const exps = col.map((v) => exp(v - maxVal));
            const sumExp = exps.reduce((a, b) => a + b, 0);
            return exps.map((e) => e / sumExp);
        });

        // Transpose back to original shape
        return Array.from({ length: numRows }, (_, i) => transposed.map((row) => row[i]));
    } else if (dim === 1) {
        // Row-wise softmax
        return matrix.map((row) => {
            const maxVal = Math.max(...row);
            const exps = row.map((v) => exp(v - maxVal));
            const sumExp = exps.reduce((a, b) => a + b, 0);
            return exps.map((e) => e / sumExp);
        });
    } else {
        throw new Error("Invalid dimension. Use 0 for columns or 1 for rows.");
    }
}

function matmul(A, B) {
    const aRows = A.length;
    const aCols = A[0].length;
    const bRows = B.length;
    const bCols = B[0].length;

    if (aCols !== bRows) {
        throw new Error("Matrix dimensions do not align for multiplication.");
    }

    const result = Array.from({ length: aRows }, () => Array.from({ length: bCols }, () => 0));

    for (let i = 0; i < aRows; i++) {
        for (let j = 0; j < bCols; j++) {
            for (let k = 0; k < aCols; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

function weightedLogicOutput(table_output, table_probs) {
    const height = table_output[0].length;
    const width = table_output[0][0].length;

    const output = Array.from({ length: height }, () =>
        Array(width).fill(0)
    );
    
    for (let i = 0; i < 16; i++) {
        const prob = table_probs[i];
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                output[y][x] += table_output[i][y][x] * prob[y];
               
            }
        }
    }

    return output; // Shape: [H][W]
}

function transpose(matrix) {
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}