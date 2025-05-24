import React, { useState } from 'react'
import FilterGroup from './FilterGroup';
import ChampionshipTable from './ChampionshipTable';
import MatchesSection from './MatchesSection';
import Header from './Header';

const FootballPredictionApp = () => {
  const [championshipLeague, setChampionshipLeague] = useState('premierleague');
  const [championshipSearch, setChampionshipSearch] = useState('');
  const [matchesLeague, setMatchesLeague] = useState('premierleague');
  const [selectedRound, setSelectedRound] = useState('all');
  const [matchesSearch, setMatchesSearch] = useState('');

  const leagueOptions = [
    { value: 'premierleague', label: 'Premier League' },
    { value: 'laliga', label: 'La Liga' },
    { value: 'bundesliga', label: 'Bundesliga' },
    { value: 'seriea', label: 'Serie A' },
    { value: 'ligue1', label: 'Ligue 1' }
  ];

  const roundOptions = [
    { value: 'all', label: 'Tất cả vòng đấu' },
    ...Array.from({ length: 38 }, (_, i) => ({
      value: i + 1,
      label: `Vòng ${i + 1}`
    }))
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-400 via-purple-500 to-purple-700 p-5">
      <div className="max-w-7xl mx-auto bg-white bg-opacity-95 rounded-3xl p-8 shadow-2xl">
        <Header />
        
        {/* Championship Section */}
        <div className="mb-12 bg-white rounded-2xl p-6 shadow-lg">
          <h2 className="text-blue-900 text-3xl font-bold mb-5 pb-3 border-b-4 border-blue-500 inline-block">
            🏆 Xác suất vô địch
          </h2>
          
          <div className="flex flex-wrap gap-5 mb-6 items-end">
            <FilterGroup label="Giải đấu:">
              <select 
                value={championshipLeague}
                onChange={(e) => setChampionshipLeague(e.target.value)}
                className="p-3 border-2 border-gray-200 rounded-lg text-base transition-all duration-300 focus:outline-none focus:border-blue-500 focus:shadow-lg min-w-52"
              >
                {leagueOptions.map(option => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </FilterGroup>
            
            <FilterGroup label="Tìm kiếm CLB:">
              <input
                type="text"
                value={championshipSearch}
                onChange={(e) => setChampionshipSearch(e.target.value)}
                placeholder="Nhập tên câu lạc bộ..."
                className="p-3 border-2 border-gray-200 rounded-lg text-base transition-all duration-300 focus:outline-none focus:border-blue-500 focus:shadow-lg min-w-52"
              />
            </FilterGroup>
          </div>

          <ChampionshipTable
            league={championshipLeague} 
            searchTerm={championshipSearch}
          />
        </div>

        {/* Matches Section */}
        <div className="bg-white rounded-2xl p-6 shadow-lg">
          <h2 className="text-blue-900 text-3xl font-bold mb-5 pb-3 border-b-4 border-blue-500 inline-block">
            📅 Dự đoán kết quả từng vòng đấu
          </h2>
          
          <div className="flex flex-wrap gap-5 mb-6 items-end">
            <FilterGroup label="Giải đấu:">
              <select 
                value={matchesLeague}
                onChange={(e) => setMatchesLeague(e.target.value)}
                className="p-3 border-2 border-gray-200 rounded-lg text-base transition-all duration-300 focus:outline-none focus:border-blue-500 focus:shadow-lg min-w-52"
              >
                {leagueOptions.map(option => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </FilterGroup>
            
            <FilterGroup label="Vòng đấu:">
              <select 
                value={selectedRound}
                onChange={(e) => setSelectedRound(e.target.value)}
                className="p-3 border-2 border-gray-200 rounded-lg text-base transition-all duration-300 focus:outline-none focus:border-blue-500 focus:shadow-lg min-w-52"
              >
                {roundOptions.map(option => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </FilterGroup>
            
            <FilterGroup label="Tìm kiếm CLB:">
              <input
                type="text"
                value={matchesSearch}
                onChange={(e) => setMatchesSearch(e.target.value)}
                placeholder="Nhập tên câu lạc bộ..."
                className="p-3 border-2 border-gray-200 rounded-lg text-base transition-all duration-300 focus:outline-none focus:border-blue-500 focus:shadow-lg min-w-52"
              />
            </FilterGroup>
          </div>

          <MatchesSection
            league={matchesLeague}
            selectedRound={selectedRound}
            searchTerm={matchesSearch}
          />
        </div>
      </div>
    </div>
  );
};

export default FootballPredictionApp