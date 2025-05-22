import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Box, Paper, TextField, Button, Typography, FormControl, InputLabel, Select, MenuItem } from '@mui/material'
import axios from 'axios'

const Login = () => {
  const navigate = useNavigate()
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    role: 'patient'
  })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  
  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
  }
  
  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    
    try {
      // For now, this is a mock authentication since the backend isn't fully implemented
      // In production, this would call the actual API endpoint
      console.log('Login attempted with:', formData)
      
      // Simulate API call
      // const response = await axios.post('/api/auth/login', formData)
      // localStorage.setItem('token', response.data.access_token)
      
      // For demo purposes, just redirect to home
      setTimeout(() => {
        navigate('/')
      }, 1000)
      
    } catch (error) {
      console.error('Login error:', error)
      setError('Invalid credentials. Please try again.')
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <Box sx={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center',
      minHeight: '80vh'
    }}>
      <Paper elevation={3} sx={{ p: 4, maxWidth: 400, width: '100%' }}>
        <Typography variant="h5" component="h1" gutterBottom align="center">
          Login to Elara AI
        </Typography>
        
        {error && (
          <Typography color="error" align="center" sx={{ mb: 2 }}>
            {error}
          </Typography>
        )}
        
        <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
          <TextField
            margin="normal"
            required
            fullWidth
            id="email"
            label="Email Address"
            name="email"
            autoComplete="email"
            autoFocus
            value={formData.email}
            onChange={handleChange}
          />
          
          <TextField
            margin="normal"
            required
            fullWidth
            name="password"
            label="Password"
            type="password"
            id="password"
            autoComplete="current-password"
            value={formData.password}
            onChange={handleChange}
          />
          
          <FormControl fullWidth margin="normal">
            <InputLabel id="role-label">User Role</InputLabel>
            <Select
              labelId="role-label"
              id="role"
              name="role"
              value={formData.role}
              label="User Role"
              onChange={handleChange}
            >
              <MenuItem value="patient">Patient</MenuItem>
              <MenuItem value="doctor">Doctor</MenuItem>
              <MenuItem value="researcher">Researcher</MenuItem>
              <MenuItem value="admin">Administrator</MenuItem>
            </Select>
          </FormControl>
          
          <Button
            type="submit"
            fullWidth
            variant="contained"
            sx={{ mt: 3, mb: 2 }}
            disabled={loading}
          >
            {loading ? 'Logging in...' : 'Login'}
          </Button>
          
          <Typography variant="body2" align="center" sx={{ mt: 2 }}>
            {"Don't have an account? Contact your administrator"}
          </Typography>
        </Box>
      </Paper>
    </Box>
  )
}

export default Login
