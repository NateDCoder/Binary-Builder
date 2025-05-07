function tableOpperator(A, B) {
    const height = A.length;
    const width = A[0].length;
    // Initialize the 3D result array: [16][height][width]
    const result = Array.from({ length: 16 }, () =>
        Array.from({ length: height }, () =>
            Array(width).fill(0)
        )
    );
    
    for (let i = 0; i < height; i++) {
        for (let j = 0; j < width; j++) {
            const a = A[i][j];
            const b = B[i][j];

            result[0][i][j]  = 0;                       // False
            result[1][i][j]  = a * b;                   // A ∧ B
            result[2][i][j]  = a - a * b;               // ¬(A ⇒ B)
            result[3][i][j]  = a;                       // A
            result[4][i][j]  = b - a * b;               // ¬(A ⇐ B)
            result[5][i][j]  = b;                       // B
            result[6][i][j]  = a + b - 2 * a * b;       // A ⊕ B
            result[7][i][j]  = a + b - a * b;           // A ∨ B
            result[8][i][j]  = 1 - (a + b - a * b);     // ¬(A ∨ B)
            result[9][i][j]  = 1 - (a + b - 2 * a * b); // ¬(A ⊕ B)
            result[10][i][j] = 1 - b;                   // ¬B
            result[11][i][j] = 1 - b + a * b;           // A ⇐ B
            result[12][i][j] = 1 - a;                   // ¬A
            result[13][i][j] = 1 - a + a * b;           // A ⇒ B
            result[14][i][j] = 1 - a * b;               // ¬(A ∧ B)
            result[15][i][j] = 1;                       // True
        }
    }

    return result;
}


function derivativeA(A, B, index) {
    const derivativesA = [
        () => 0,         // 0: False
        () => B,         // 1: A ∧ B
        () => 1 - B,     // 2: ¬(A ⇒ B)
        () => 1,         // 3: A
        () => -B,        // 4: ¬(A ⇐ B)
        () => 0,         // 5: B
        () => 1 - 2 * B, // 6: A ⊕ B
        () => 1 - B,     // 7: A ∨ B
        () => B - 1,     // 8: ¬(A ∨ B)
        () => 2 * B - 1, // 9: ¬(A ⊕ B)
        () => 0,         // 10: ¬B
        () => B,         // 11: A ⇐ B
        () => -1,        // 12: ¬A
        () => -1 + B,    // 13: A ⇒ B
        () => -B,        // 14: ¬(A ∧ B)
        () => 0          // 15: True
    ];
    return derivativesA[index]();
}
function derivativeB(A, B, index) {
    const derivativesB = [
        () => 0,         // 0: False
        () => A,         // 1: A ∧ B
        () => -A,        // 2: ¬(A ⇒ B)
        () => 0,         // 3: A
        () => 1 - A,     // 4: ¬(A ⇐ B)
        () => 1,         // 5: B
        () => 1 - 2 * A, // 6: A ⊕ B
        () => 1 - A,     // 7: A ∨ B
        () => A - 1,     // 8: ¬(A ∨ B)
        () => 2 * A - 1, // 9: ¬(A ⊕ B)
        () => -1,        // 10: ¬B
        () => -1 + A,    // 11: A ⇐ B
        () => 0,         // 12: ¬A
        () => A,         // 13: A ⇒ B
        () => -A,        // 14: ¬(A ∧ B)
        () => 0          // 15: True
    ];
    return derivativesB[index]();
}

