import React from 'react'

const MatchPrediction = ({ prediction }) => {
  return (
    <div className="text-center p-2 rounded-lg bg-white shadow-md min-w-16">
      <div className="text-xs text-gray-600 mb-1">{prediction.label}</div>
      <div className="font-bold text-blue-900">{prediction.value}%</div>
    </div>
  );
};

export default MatchPrediction