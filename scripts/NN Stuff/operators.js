function tableOpperator(A, B, index) {
    const operations = [
        () => 0,                       // 0: False
        () => A * B,                   // 1: A ∧ B
        () => A - A * B,               // 2: ¬(A ⇒ B)
        () => A,                       // 3: A
        () => B - A * B,               // 4: ¬(A ⇐ B)
        () => B,                       // 5: B
        () => A + B - 2 * A * B,       // 6: A ⊕ B
        () => A + B - A * B,           // 7: A ∨ B
        () => 1 - (A + B - A * B),     // 8: ¬(A ∨ B)
        () => 1 - (A + B - 2 * A * B), // 9: ¬(A ⊕ B)
        () => 1 - B,                   // 10: ¬B
        () => 1 - B + A * B,           // 11: A ⇐ B
        () => 1 - A,                   // 12: ¬A
        () => 1 - A + A * B,           // 13: A ⇒ B
        () => 1 - A * B,               // 14: ¬(A ∧ B)
        () => 1                        // 15: True
    ];
    return operations[index]();
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
