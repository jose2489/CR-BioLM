import { useState } from "react";

export default function InputPanel({ startJob }) {
  const [species, setSpecies] = useState("");
  const [question, setQuestion] = useState("");

  return (
    <div className="bg-white p-6 rounded-xl shadow">
      <input
        className="border p-2 w-full mb-2"
        placeholder="Quercus costaricensis"
        onChange={(e) => setSpecies(e.target.value)}
      />
      <textarea
        className="border p-2 w-full mb-2"
        placeholder="Pregunta..."
        onChange={(e) => setQuestion(e.target.value)}
      />
      <button
        onClick={() => startJob(species, question)}
        className="bg-green-600 text-white px-4 py-2 rounded"
      >
        Ejecutar
      </button>
    </div>
  );
}