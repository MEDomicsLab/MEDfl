echo '#!/bin/bash

# Update package lists
sudo apt update

# Install MySQL
sudo apt install mysql-server

# Secure MySQL installation
sudo mysql_secure_installation

# Install phpMyAdmin
sudo apt install phpmyadmin

# Create symbolic link for Apache
sudo ln -s /etc/phpmyadmin/apache.conf /etc/apache2/conf-available/phpmyadmin.conf
sudo a2enconf phpmyadmin
sudo systemctl reload apache2

# Print completion message
echo "MySQL and phpMyAdmin setup complete."
' > setup_mysql.sh && chmod +x setup_mysql.sh && python3 scripts/create_db.py
