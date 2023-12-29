// App.js

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css'; // Import the CSS file

function App() {
  const [data, setData] = useState([]);

  useEffect(() => {
    // Fetch data from the server
    axios.get('http://127.0.0.1:5000/get_data')
      .then(response => {
        setData(response.data);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });
  }, []);

  return (
    <div className="app-container">
      <h1>Frame Data Table</h1>
      <table className="data-table">
        <thead>
          <tr>
            <th>Frame Number</th>
            <th>Class Label</th>
            <th>Confidence</th>
            <th>X Midpoint</th>
            <th>Y Midpoint</th>
            <th>Width</th>
            <th>Height</th>
          </tr>
        </thead>
        <tbody>
          {data.map(frame => (
            <tr key={frame.frameNumber}>
              <td>{frame.frameNumber}</td>
              <td>{frame.classLabel}</td>
              <td>{frame.confidence}</td>
              <td>{frame.xMidpoint}</td>
              <td>{frame.yMidpoint}</td>
              <td>{frame.width}</td>
              <td>{frame.height}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
