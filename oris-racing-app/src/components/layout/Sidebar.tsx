import React from 'react';
import { 
  LayoutDashboard, 
  Activity, 
  Target, 
  Timer, 
  GraduationCap, 
  History,
  Settings,
  HelpCircle,
  Map,
  Wrench
} from 'lucide-react';
import styles from './Sidebar.module.css';

interface NavItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  description?: string;
}

interface SidebarProps {
  activeItem?: string;
  onItemClick?: (itemId: string) => void;
}

const navigationItems: NavItem[] = [
  { 
    id: 'dashboard', 
    label: 'Dashboard', 
    icon: <LayoutDashboard size={18} />,
    description: 'Real-time race overview'
  },
  { 
    id: 'telemetry', 
    label: 'Telemetry', 
    icon: <Activity size={18} />,
    description: 'Live vehicle data streams'
  },
  { 
    id: 'strategy', 
    label: 'Strategy', 
    icon: <Target size={18} />,
    description: 'Race strategy planning'
  },
  { 
    id: 'timing', 
    label: 'Timing', 
    icon: <Timer size={18} />,
    description: 'Lap times and sector analysis'
  },
  { 
    id: 'training', 
    label: 'Training', 
    icon: <GraduationCap size={18} />,
    description: 'Driver performance analysis'
  },
  { 
    id: 'history', 
    label: 'History', 
    icon: <History size={18} />,
    description: 'Past race data'
  },
  { 
    id: 'tracks', 
    label: 'Tracks', 
    icon: <Map size={18} />,
    description: 'Track information and layouts'
  },
  { 
    id: 'garage', 
    label: 'Garage', 
    icon: <Wrench size={18} />,
    description: 'Vehicle setup and tuning'
  },
];

export const Sidebar: React.FC<SidebarProps> = ({ 
  activeItem = 'dashboard',
  onItemClick 
}) => {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.header}>
        <h3 className={styles.headerTitle}>Navigation</h3>
      </div>

      <nav className={styles.nav}>
        {navigationItems.map((item) => (
          <button
            key={item.id}
            className={`${styles.navItem} ${activeItem === item.id ? styles.active : ''}`}
            onClick={() => onItemClick?.(item.id)}
            title={item.description}
            style={{
              background: activeItem === item.id 
                ? 'linear-gradient(to right, rgba(20, 184, 166, 0.1), rgba(6, 182, 212, 0.1))'
                : 'transparent'
            }}
          >
            {activeItem === item.id && <div className={styles.activeIndicator} />}
            <span className={styles.icon}>{item.icon}</span>
            <span className={styles.label}>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className={styles.bottomSection}>
        <button 
          className={styles.navItem}
          onClick={() => onItemClick?.('settings')}
        >
          <span className={styles.icon}><Settings size={18} /></span>
          <span className={styles.label}>Settings</span>
        </button>
        
        <button 
          className={styles.navItem}
          onClick={() => onItemClick?.('help')}
        >
          <span className={styles.icon}><HelpCircle size={18} /></span>
          <span className={styles.label}>Help</span>
        </button>
      </div>
      
      <div className={styles.footer}>
        <div className={styles.connectionStatus}>
          <div className={styles.statusDot} />
          <span className={styles.statusText}>Live Connection</span>
        </div>
        <div className={styles.version}>
          ORIS v1.0.0
        </div>
      </div>
    </aside>
  );
};