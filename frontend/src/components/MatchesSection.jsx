import React, { useEffect, useState } from 'react'
import MatchRow from './MatchRow';

const MatchesSection = ({ league, selectedRound, searchTerm }) => {
  const [matchesData, setMatchesData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filteredRounds, setFilteredRounds] = useState([]);

  useEffect(() => {
    const fetchMatches = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch("http://localhost:8000/matches/predictions?model=xgboost");
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setMatchesData(data.data || {});
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchMatches();
  }, []);

  useEffect(() => {
    console.log(matchesData)
  }, [matchesData])

  useEffect(() => {
    if(matchesData && matchesData[league]){
      const rounds = Object.entries(matchesData[league].matches_by_round || {});
      setFilteredRounds(rounds)
    } else {
      setFilteredRounds([])
    }
  }, [matchesData, league]);

  const displayedRounds = React.useMemo(() => {
    let rounds = filteredRounds;

    if (selectedRound !== 'all') {
      rounds = rounds.filter(([round]) => parseInt(round) === parseInt(selectedRound));
    }

    if (searchTerm) {
      rounds = rounds
        .map(([round, matches]) => [
          round,
          matches.filter(
            (match) =>
              match.home_team.toLowerCase().includes(searchTerm.toLowerCase()) ||
              match.away_team.toLowerCase().includes(searchTerm.toLowerCase())
          ),
        ])
        .filter(([_, matches]) => matches.length > 0);
    }

    return rounds;
  }, [filteredRounds, selectedRound, searchTerm]);

  if (loading) return <div>Loading matches...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      {displayedRounds.length === 0 ? (
        <div>No matches found.</div>
      ) : (
        displayedRounds
          .sort((a, b) => a[0] - b[0])
          .map(([roundNum, matches]) => (
            <div key={roundNum}>
              <div className="bg-gradient-to-r from-purple-600 to-purple-700 text-white p-4 my-5 rounded-lg font-semibold text-center text-lg">
                <span className="inline-block px-4 py-1 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full text-sm mr-3">
                  {matchesData.league_name}
                </span>
                Vòng {roundNum}
              </div>

              {matches.map((match, index) => (
                <MatchRow key={`${match.home_team}-${match.away_team}-${index}`} match={match} />
              ))}
            </div>
          ))
      )}
    </div>
  );
};

export default MatchesSection
