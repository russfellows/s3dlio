#!/usr/bin/env bash
# install-system-deps.sh
#
# Install system dependencies required for building s3dlio
#
# SPDX-License-Identifier: Apache-2.0 OR MIT
# SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}s3dlio System Dependencies Installer${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Detect OS and distribution
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$ID
            VER=$VERSION_ID
        elif type lsb_release >/dev/null 2>&1; then
            OS=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
            VER=$(lsb_release -sr)
        else
            OS="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        OS="unknown"
    fi
    
    echo -e "${GREEN}Detected OS: ${OS}${NC}"
}

# Install dependencies based on OS
install_deps() {
    case "$OS" in
        ubuntu|debian|pop)
            echo -e "${YELLOW}Installing dependencies for Ubuntu/Debian...${NC}"
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                pkg-config \
                libssl-dev \
                libhdf5-dev \
                libhwloc-dev \
                cmake \
                curl \
                git
            ;;
        
        rhel|centos|fedora|rocky|almalinux)
            echo -e "${YELLOW}Installing dependencies for RHEL/CentOS/Fedora...${NC}"
            if command -v dnf &> /dev/null; then
                PKG_MGR="dnf"
            else
                PKG_MGR="yum"
            fi
            
            sudo $PKG_MGR install -y \
                gcc \
                gcc-c++ \
                make \
                pkg-config \
                openssl-devel \
                hdf5-devel \
                hwloc-devel \
                cmake \
                curl \
                git
            ;;
        
        arch|manjaro)
            echo -e "${YELLOW}Installing dependencies for Arch Linux...${NC}"
            sudo pacman -Sy --noconfirm \
                base-devel \
                pkg-config \
                openssl \
                hdf5 \
                hwloc \
                cmake \
                curl \
                git
            ;;
        
        opensuse*|sles)
            echo -e "${YELLOW}Installing dependencies for openSUSE/SLES...${NC}"
            sudo zypper install -y \
                gcc \
                gcc-c++ \
                make \
                pkg-config \
                libopenssl-devel \
                hdf5-devel \
                hwloc-devel \
                cmake \
                curl \
                git
            ;;
        
        macos)
            echo -e "${YELLOW}Installing dependencies for macOS...${NC}"
            if ! command -v brew &> /dev/null; then
                echo -e "${RED}Error: Homebrew not found. Please install from https://brew.sh${NC}"
                exit 1
            fi
            
            brew install \
                pkg-config \
                openssl@3 \
                hdf5 \
                hwloc \
                cmake
            
            # Add OpenSSL to PKG_CONFIG_PATH for Homebrew
            echo ""
            echo -e "${YELLOW}Note: You may need to set these environment variables:${NC}"
            echo -e "export PKG_CONFIG_PATH=\"$(brew --prefix openssl@3)/lib/pkgconfig:\$PKG_CONFIG_PATH\""
            echo -e "export OPENSSL_DIR=\"$(brew --prefix openssl@3)\""
            ;;
        
        *)
            echo -e "${RED}Unsupported OS: $OS${NC}"
            echo -e "${YELLOW}Please install the following dependencies manually:${NC}"
            echo "  - C/C++ compiler (gcc/clang)"
            echo "  - pkg-config"
            echo "  - OpenSSL development libraries"
            echo "  - HDF5 development libraries"
            echo "  - hwloc development libraries (optional, for NUMA support)"
            echo "  - CMake"
            echo "  - curl"
            echo "  - git"
            exit 1
            ;;
    esac
}

# Check if Rust is installed
check_rust() {
    if command -v rustc &> /dev/null; then
        RUST_VERSION=$(rustc --version)
        echo -e "${GREEN}✓ Rust is installed: ${RUST_VERSION}${NC}"
    else
        echo -e "${YELLOW}⚠ Rust is not installed${NC}"
        echo -e "${YELLOW}Install Rust from: https://rustup.rs/${NC}"
        echo -e "Run: ${BLUE}curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh${NC}"
        NEED_RUST=1
    fi
}

# Verify installation
verify_deps() {
    echo ""
    echo -e "${BLUE}Verifying dependencies...${NC}"
    
    MISSING=0
    
    # Check pkg-config
    if command -v pkg-config &> /dev/null; then
        echo -e "${GREEN}✓ pkg-config${NC}"
    else
        echo -e "${RED}✗ pkg-config not found${NC}"
        MISSING=1
    fi
    
    # Check HDF5
    if pkg-config --exists hdf5 2>/dev/null; then
        HDF5_VERSION=$(pkg-config --modversion hdf5)
        echo -e "${GREEN}✓ HDF5 ${HDF5_VERSION}${NC}"
    else
        echo -e "${YELLOW}⚠ HDF5 not found via pkg-config (may still work)${NC}"
    fi
    
    # Check hwloc (optional)
    if pkg-config --exists hwloc 2>/dev/null; then
        HWLOC_VERSION=$(pkg-config --modversion hwloc)
        echo -e "${GREEN}✓ hwloc ${HWLOC_VERSION} (NUMA support enabled)${NC}"
    else
        echo -e "${YELLOW}⚠ hwloc not found (NUMA support disabled)${NC}"
    fi
    
    # Check OpenSSL
    if pkg-config --exists openssl 2>/dev/null; then
        OPENSSL_VERSION=$(pkg-config --modversion openssl)
        echo -e "${GREEN}✓ OpenSSL ${OPENSSL_VERSION}${NC}"
    else
        echo -e "${RED}✗ OpenSSL not found${NC}"
        MISSING=1
    fi
    
    # Check compiler
    if command -v gcc &> /dev/null || command -v clang &> /dev/null; then
        echo -e "${GREEN}✓ C/C++ compiler${NC}"
    else
        echo -e "${RED}✗ C/C++ compiler not found${NC}"
        MISSING=1
    fi
    
    # Check CMake
    if command -v cmake &> /dev/null; then
        CMAKE_VERSION=$(cmake --version | head -n1)
        echo -e "${GREEN}✓ ${CMAKE_VERSION}${NC}"
    else
        echo -e "${YELLOW}⚠ CMake not found (may be needed for some dependencies)${NC}"
    fi
    
    echo ""
    
    if [ $MISSING -eq 0 ]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}✓ All required dependencies installed!${NC}"
        echo -e "${GREEN}========================================${NC}"
        
        if [ -z "$NEED_RUST" ]; then
            echo ""
            echo -e "${BLUE}Next steps:${NC}"
            echo -e "  ${BLUE}1.${NC} Build s3dlio:"
            echo -e "     ${BLUE}cargo build --release${NC}"
            echo -e "  ${BLUE}2.${NC} Run tests:"
            echo -e "     ${BLUE}cargo test${NC}"
            echo -e "  ${BLUE}3.${NC} Build with all features:"
            echo -e "     ${BLUE}cargo build --release --all-features${NC}"
        fi
    else
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}Some dependencies are missing!${NC}"
        echo -e "${RED}========================================${NC}"
        echo -e "${YELLOW}Please check the errors above and install missing packages.${NC}"
        exit 1
    fi
}

# Main execution
main() {
    detect_os
    echo ""
    
    # Ask for confirmation unless --yes flag is provided
    if [[ "$1" != "--yes" ]] && [[ "$1" != "-y" ]]; then
        read -p "Install system dependencies? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Installation cancelled.${NC}"
            exit 0
        fi
    fi
    
    install_deps
    echo ""
    check_rust
    echo ""
    verify_deps
}

# Show help
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Install system dependencies required for building s3dlio"
    echo ""
    echo "Options:"
    echo "  -y, --yes     Skip confirmation prompt"
    echo "  -h, --help    Show this help message"
    echo ""
    echo "Supported platforms:"
    echo "  - Ubuntu/Debian"
    echo "  - RHEL/CentOS/Fedora/Rocky/AlmaLinux"
    echo "  - Arch Linux/Manjaro"
    echo "  - openSUSE/SLES"
    echo "  - macOS (via Homebrew)"
    echo ""
    echo "Dependencies installed:"
    echo "  - build-essential/gcc/clang (C/C++ compiler)"
    echo "  - pkg-config"
    echo "  - OpenSSL development libraries"
    echo "  - HDF5 development libraries"
    echo "  - hwloc development libraries (for NUMA support)"
    echo "  - CMake"
    echo "  - curl, git"
    exit 0
fi

main "$@"
