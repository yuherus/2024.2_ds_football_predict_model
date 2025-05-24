import React from 'react'

const FilterGroup = ({ label, children }) => {
  return (
    <div className="flex flex-col gap-1">
      <label className="font-semibold text-blue-900 text-sm">{label}</label>
      {children}
    </div>
  );
};

export default FilterGroup