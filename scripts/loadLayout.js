// function loadProgress() {
//     const savedState = localStorage.getItem("circuitState");
//     if (savedState) {
//       const state = JSON.parse(savedState);
  
//       // A function to restore the suppliers (you can adjust this logic as needed)
//       const restoreSupplierFunc = (value) => {
//         return value; // This simply returns the saved value, but you can customize it further.
//       };
  
//       // Restore logic gates
//       logicGates = state.logicGates.map(data => {
//         const gate = new LogicGate(data.x, data.y, data.type);
//         if (data.inputALine) gate.inputALine = Line.fromJSON(data.inputALine, restoreSupplierFunc);
//         if (data.inputBLine) gate.inputBLine = Line.fromJSON(data.inputBLine, restoreSupplierFunc);
//         return gate;
//       });
  
//       // Restore lines
//       lines = state.lines.map(lineData => Line.fromJSON(lineData, restoreSupplierFunc));
  
//       // Restore binary displays
//       currentIndex = state.currentIndex;
//       number1.binary = state.number1Binary;
//       answer.binary = state.answerBinary;
  
//       console.log("Progress loaded.");
//     } else {
//       console.log("No saved progress found.");
//     }
//   }
  