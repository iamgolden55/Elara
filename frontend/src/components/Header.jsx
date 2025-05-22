import { useState } from 'react'
import { Link } from 'react-router-dom'
import { AppBar, Toolbar, Typography, Button, IconButton, Menu, MenuItem, Box } from '@mui/material'
import Brightness4Icon from '@mui/icons-material/Brightness4'
import Brightness7Icon from '@mui/icons-material/Brightness7'
import LanguageIcon from '@mui/icons-material/Language'
import AccountCircleIcon from '@mui/icons-material/AccountCircle'
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety'

const Header = ({ darkMode, setDarkMode }) => {
  const [languageAnchorEl, setLanguageAnchorEl] = useState(null)
  const [userAnchorEl, setUserAnchorEl] = useState(null)
  
  const languageMenuOpen = Boolean(languageAnchorEl)
  const userMenuOpen = Boolean(userAnchorEl)
  
  const handleLanguageMenuClick = (event) => {
    setLanguageAnchorEl(event.currentTarget)
  }
  
  const handleUserMenuClick = (event) => {
    setUserAnchorEl(event.currentTarget)
  }
  
  const handleLanguageMenuClose = () => {
    setLanguageAnchorEl(null)
  }
  
  const handleUserMenuClose = () => {
    setUserAnchorEl(null)
  }
  
  const handleLanguageSelect = (language) => {
    // Add language selection functionality here
    console.log(`Selected language: ${language}`)
    handleLanguageMenuClose()
  }
  
  return (
    <AppBar position="static">
      <Toolbar>
        <HealthAndSafetyIcon sx={{ mr: 1 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Elara AI
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {/* Language Menu */}
          <IconButton 
            color="inherit" 
            onClick={handleLanguageMenuClick}
            aria-controls={languageMenuOpen ? 'language-menu' : undefined}
            aria-haspopup="true"
            aria-expanded={languageMenuOpen ? 'true' : undefined}
          >
            <LanguageIcon />
          </IconButton>
          <Menu
            id="language-menu"
            anchorEl={languageAnchorEl}
            open={languageMenuOpen}
            onClose={handleLanguageMenuClose}
          >
            <MenuItem onClick={() => handleLanguageSelect('en')}>English</MenuItem>
            <MenuItem onClick={() => handleLanguageSelect('es')}>Español</MenuItem>
            <MenuItem onClick={() => handleLanguageSelect('fr')}>Français</MenuItem>
            <MenuItem onClick={() => handleLanguageSelect('de')}>Deutsch</MenuItem>
            <MenuItem onClick={() => handleLanguageSelect('zh')}>中文</MenuItem>
          </Menu>
          
          {/* Theme Toggle */}
          <IconButton color="inherit" onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? <Brightness7Icon /> : <Brightness4Icon />}
          </IconButton>
          
          {/* User Menu */}
          <IconButton 
            color="inherit" 
            onClick={handleUserMenuClick}
            aria-controls={userMenuOpen ? 'user-menu' : undefined}
            aria-haspopup="true"
            aria-expanded={userMenuOpen ? 'true' : undefined}
          >
            <AccountCircleIcon />
          </IconButton>
          <Menu
            id="user-menu"
            anchorEl={userAnchorEl}
            open={userMenuOpen}
            onClose={handleUserMenuClose}
          >
            <MenuItem onClick={handleUserMenuClose} component={Link} to="/login">
              Login
            </MenuItem>
            <MenuItem onClick={handleUserMenuClose}>
              Profile
            </MenuItem>
            <MenuItem onClick={handleUserMenuClose}>
              Logout
            </MenuItem>
          </Menu>
        </Box>
      </Toolbar>
    </AppBar>
  )
}

export default Header
