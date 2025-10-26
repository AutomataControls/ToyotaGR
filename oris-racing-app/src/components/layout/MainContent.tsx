import React from 'react';
import styles from './MainContent.module.css';

interface MainContentProps {
  children: React.ReactNode;
}

export const MainContent: React.FC<MainContentProps> = ({ children }) => {
  return (
    <main className={styles.main}>
      {children}
    </main>
  );
};