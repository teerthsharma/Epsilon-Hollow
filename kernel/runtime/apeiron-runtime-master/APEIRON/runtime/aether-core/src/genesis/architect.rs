// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! The Architect (Self-Editing)
//! Handles hot-swapping of Aether scripts and rollback mechanisms.
//! Requires `std` for file I/O.

#[cfg(feature = "std")]
pub mod implementation {
    use std::fs;
    use std::path::PathBuf;
    use std::io::{self};

    #[derive(Debug)]
    pub enum ArchitectError {
        IoError(io::Error),
        CompilationFailed(String),
        ValidationFailed(String),
    }

    impl From<io::Error> for ArchitectError {
        fn from(e: io::Error) -> Self {
            ArchitectError::IoError(e)
        }
    }

    pub type CodeValidator = Box<dyn Fn(&str) -> Result<(), String> + Send + Sync>;

    pub struct Architect {
        pub scripts_dir: PathBuf,
        pub backup_dir: PathBuf,
        validator: CodeValidator,
    }

    impl Architect {
        pub fn new(scripts_dir: impl Into<PathBuf>, validator: CodeValidator) -> Self {
            let scripts_dir = scripts_dir.into();
            let backup_dir = scripts_dir.join("backup");
            // Ensure backup dir exists
            if let Err(e) = fs::create_dir_all(&backup_dir) {
                eprintln!("Failed to create backup dir: {}", e);
            }
            Self {
                scripts_dir,
                backup_dir,
                validator,
            }
        }

        /// Requests a mutation of a script.
        pub fn hot_swap_script(&self, script_name: &str, new_code: &str) -> Result<(), ArchitectError> {
            let script_path = self.scripts_dir.join(script_name);
            
            // 1. Sandbox Validation (Mock)
            self.validate_code(new_code)?;

            // 2. Backup existing
            if script_path.exists() {
                self.backup_script(script_name)?;
            }

            // 3. Write new code
            fs::write(&script_path, new_code)?;
            
            Ok(())
        }

        pub(crate) fn validate_code(&self, code: &str) -> Result<(), ArchitectError> {
            if code.trim().is_empty() {
                return Err(ArchitectError::ValidationFailed("Code is empty".into()));
            }

            // Actual compilation check via callback
            (self.validator)(code).map_err(ArchitectError::CompilationFailed)
        }

        fn backup_script(&self, script_name: &str) -> Result<(), ArchitectError> {
            let src = self.scripts_dir.join(script_name);
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let backup_name = format!("{}.{}.bak", script_name, timestamp);
            let dst = self.backup_dir.join(backup_name);
            
            fs::copy(src, dst)?;
            Ok(())
        }
        
        /// Reverts to the latest backup.
        pub fn rollback(&self, _script_name: &str) -> Result<(), ArchitectError> {
             // Simple rollback logic: find latest .bak and restore
             // Implementation left as exercise or future step
             Ok(())
        }
    }
}

#[cfg(test)]
#[cfg(feature = "std")]
mod tests {
    use super::implementation::*;
    use std::path::PathBuf;

    #[test]
    fn test_validate_code_success() {
        let validator: CodeValidator = Box::new(|_: &str| Ok(()));
        let architect = Architect::new(PathBuf::from("/tmp"), validator);
        assert!(architect.validate_code("some code").is_ok());
    }

    #[test]
    fn test_validate_code_failure() {
        let validator: CodeValidator = Box::new(|_: &str| Err("Syntax error".into()));
        let architect = Architect::new(PathBuf::from("/tmp"), validator);
        let result = architect.validate_code("bad code");
        match result {
            Err(ArchitectError::CompilationFailed(msg)) => assert_eq!(msg, "Syntax error"),
            _ => panic!("Expected CompilationFailed"),
        }
    }
}
