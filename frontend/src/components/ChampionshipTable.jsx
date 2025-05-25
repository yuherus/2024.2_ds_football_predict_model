import React, { useEffect, useState } from 'react';

const ChampionshipTable = ({ league, searchTerm }) => {
  const [leagueData, setLeagueData] = useState({})
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [teams , setTeams] = useState([])
  const [filteredTeams, setFilteredTeams] = useState([])

  useEffect(() => {
    const fetchMatches = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch("http://localhost:8000/championship_probabilities");
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setLeagueData(data || {});
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchMatches();
  }, []);

  useEffect(() => {
    setTeams(leagueData[league]?.teams || [])
  }, [leagueData, league])

  useEffect(() => {
    if(searchTerm && searchTerm.lenght > 0) {
      setFilteredTeams(teams.filter(team => team.team_name.toLowerCase().includes(searchTerm.toLowerCase()) ))
    } else {
      setFilteredTeams(teams)
    }
    console.log(searchTerm)
  }, [searchTerm, teams])

  const getProbabilityClass = (prob) => {
    if (prob > 30) return 'bg-gradient-to-r from-red-500 to-red-600';
    if (prob > 5) return 'bg-gradient-to-r from-orange-500 to-orange-600';
    return 'bg-gradient-to-r from-gray-500 to-gray-600';
  };

  return (

    <div className="overflow-x-auto rounded-lg shadow-lg">
      <table className="w-full bg-white border-collapse">
        <thead>
          <tr className="bg-gradient-to-r from-blue-500 to-blue-600 text-white">
            <th className="p-4 text-left font-semibold uppercase tracking-wide text-sm">Hạng</th>
            <th className="p-4 text-left font-semibold uppercase tracking-wide text-sm">Câu lạc bộ</th>
            <th className="p-4 text-left font-semibold uppercase tracking-wide text-sm">Xác suất vô địch</th>
          </tr>
        </thead>
        <tbody>
          {filteredTeams.length === 0 ? (
              <div>No matches found.</div>
            ) : (
              filteredTeams.map((team) => (
                <tr key={team.team_name} className="hover:bg-gray-50 hover:transform hover:-translate-y-0.5 transition-all duration-200 border-b border-gray-100">
                  <td className="p-4">{team.rank}</td>
                  <td className="p-4">
                    <div className="flex items-center">
                      <span className="font-semibold text-blue-900">{team.team_name}</span>
                    </div>
                  </td>
                  <td className="p-4">
                    <span className={`px-3 py-1 rounded-full text-white font-bold text-center min-w-16 inline-block ${getProbabilityClass(team.championship_probability)}`}>
                      {(team.championship_probability * 100).toFixed(4)}%
                    </span>
                  </td>
                </tr>
              ))
            )
          }
        </tbody>
      </table>
    </div>
  );
};

export default ChampionshipTable;
