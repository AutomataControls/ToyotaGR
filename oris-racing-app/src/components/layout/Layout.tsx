import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { MainContent } from './MainContent';
import styles from './Layout.module.css';

interface LayoutProps {
  children: React.ReactNode;
  isLive?: boolean;
  sessionType?: string;
  activeNav?: string;
  onNavChange?: (itemId: string) => void;
}

export const Layout: React.FC<LayoutProps> = ({ 
  children, 
  isLive = false,
  sessionType = 'Practice',
  activeNav = 'dashboard',
  onNavChange
}) => {
  const navigate = useNavigate();

  const handleNavClick = (itemId: string) => {
    navigate(`/${itemId}`);
    onNavChange?.(itemId);
  };

  return (
    <div className={styles.layout}>
      <Header isLive={isLive} sessionType={sessionType} />
      <div className={styles.body}>
        <Sidebar activeItem={activeNav} onItemClick={handleNavClick} />
        <MainContent>{children}</MainContent>
      </div>
    </div>
  );
};