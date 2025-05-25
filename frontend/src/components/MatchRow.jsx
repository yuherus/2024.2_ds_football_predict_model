import React from 'react'
import MatchPrediction from './MatchPrediction';

const MatchRow = ({ match }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 mb-3 bg-gray-50 rounded-lg border-l-4 border-blue-500 items-center">
      <div className="flex justify-between items-center font-semibold text-blue-900 md:justify-center">
        <span>{match.home_team}</span>
        <span className="text-gray-500 text-sm mx-2">vs</span>
        <span>{match.away_team}</span>
      </div>
      
      <div className="flex gap-3 justify-center">
        <MatchPrediction prediction={{ label: 'Thắng', value: (match.predictions.home_win*100).toFixed(4) }} />
        <MatchPrediction prediction={{ label: 'Hòa', value: (match.predictions.draw*100).toFixed(4) }} />
        <MatchPrediction prediction={{ label: 'Thua', value: (match.predictions.away_win*100).toFixed(4) }} />
      </div>
      
      <div className="text-right text-gray-600 md:text-center text-sm">
        <div>{new Date(match.match_date).toLocaleDateString('vi-VN')}</div>
        <div className="italic text-xs">{match.venue}</div>
      </div>
    </div>
  );
};

export default MatchRow;
