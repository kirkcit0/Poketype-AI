import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
import numpy as np
import os
import random
from collections import defaultdict

# pokemon assigned types

GEN1_POKEMON_TYPES = {
    'Bulbasaur': ['Grass', 'Poison'],
    'Ivysaur': ['Grass', 'Poison'],
    'Venusaur': ['Grass', 'Poison'],
    'Charmander': ['Fire'],
    'Charmeleon': ['Fire'],
    'Charizard': ['Fire', 'Flying'],
    'Squirtle': ['Water'],
    'Wartortle': ['Water'],
    'Blastoise': ['Water'],
    'Caterpie': ['Bug'],
    'Metapod': ['Bug'],
    'Butterfree': ['Bug', 'Flying'],
    'Weedle': ['Bug', 'Poison'],
    'Kakuna': ['Bug', 'Poison'],
    'Beedrill': ['Bug', 'Poison'],
    'Pidgey': ['Normal', 'Flying'],
    'Pidgeotto': ['Normal', 'Flying'],
    'Pidgeot': ['Normal', 'Flying'],
    'Rattata': ['Normal'],
    'Raticate': ['Normal'],
    'Spearow': ['Normal', 'Flying'],
    'Fearow': ['Normal', 'Flying'],
    'Ekans': ['Poison'],
    'Arbok': ['Poison'],
    'Pikachu': ['Electric'],
    'Raichu': ['Electric'],
    'Sandshrew': ['Ground'],
    'Sandslash': ['Ground'],
    'Nidoran♀': ['Poison'],
    'Nidorina': ['Poison'],
    'Nidoqueen': ['Poison', 'Ground'],
    'Nidoran♂': ['Poison'],
    'Nidorino': ['Poison'],
    'Nidoking': ['Poison', 'Ground'],
    'Clefairy': ['Normal'],
    'Clefable': ['Normal'], 
    'Vulpix': ['Fire'],
    'Ninetales': ['Fire'],
    'Jigglypuff': ['Normal'],
    'Wigglytuff': ['Normal'], 
    'Zubat': ['Poison', 'Flying'],
    'Golbat': ['Poison', 'Flying'],
    'Oddish': ['Grass', 'Poison'],
    'Gloom': ['Grass', 'Poison'],
    'Vileplume': ['Grass', 'Poison'],
    'Paras': ['Bug', 'Grass'],
    'Parasect': ['Bug', 'Grass'],
    'Venonat': ['Bug', 'Poison'],
    'Venomoth': ['Bug', 'Poison'],
    'Diglett': ['Ground'],
    'Dugtrio': ['Ground'],
    'Meowth': ['Normal'],
    'Persian': ['Normal'],
    'Psyduck': ['Water'],
    'Golduck': ['Water'],
    'Mankey': ['Fighting'],
    'Primeape': ['Fighting'],
    'Growlithe': ['Fire'],
    'Arcanine': ['Fire'],
    'Poliwag': ['Water'],
    'Poliwhirl': ['Water'],
    'Poliwrath': ['Water', 'Fighting'],
    'Abra': ['Psychic'],
    'Kadabra': ['Psychic'],
    'Alakazam': ['Psychic'],
    'Machop': ['Fighting'],
    'Machoke': ['Fighting'],
    'Machamp': ['Fighting'],
    'Bellsprout': ['Grass', 'Poison'],
    'Weepinbell': ['Grass', 'Poison'],
    'Victreebel': ['Grass', 'Poison'],
    'Tentacool': ['Water', 'Poison'],
    'Tentacruel': ['Water', 'Poison'],
    'Geodude': ['Rock', 'Ground'],
    'Graveler': ['Rock', 'Ground'],
    'Golem': ['Rock', 'Ground'],
    'Ponyta': ['Fire'],
    'Rapidash': ['Fire'],
    'Slowpoke': ['Water', 'Psychic'],
    'Slowbro': ['Water', 'Psychic'],
    'Magnemite': ['Electric', 'Steel'],
    'Magneton': ['Electric', 'Steel'], 
    "Farfetchd": ['Normal', 'Flying'],
    'Doduo': ['Normal', 'Flying'],
    'Dodrio': ['Normal', 'Flying'],
    'Seel': ['Water'],
    'Dewgong': ['Water', 'Ice'],
    'Grimer': ['Poison'],
    'Muk': ['Poison'],
    'Shellder': ['Water'],
    'Cloyster': ['Water', 'Ice'],
    'Gastly': ['Ghost', 'Poison'],
    'Haunter': ['Ghost', 'Poison'],
    'Gengar': ['Ghost', 'Poison'],
    'Onix': ['Rock', 'Ground'],
    'Drowzee': ['Psychic'],
    'Hypno': ['Psychic'],
    'Krabby': ['Water'],
    'Kingler': ['Water'],
    'Voltorb': ['Electric'],
    'Electrode': ['Electric'],
    'Exeggcute': ['Grass', 'Psychic'],
    'Exeggutor': ['Grass', 'Psychic'],
    'Cubone': ['Ground'],
    'Marowak': ['Ground'],
    'Hitmonlee': ['Fighting'],
    'Hitmonchan': ['Fighting'],
    'Lickitung': ['Normal'],
    'Koffing': ['Poison'],
    'Weezing': ['Poison'],
    'Rhyhorn': ['Ground', 'Rock'],
    'Rhydon': ['Ground', 'Rock'],
    'Chansey': ['Normal'],
    'Tangela': ['Grass'],
    'Kangaskhan': ['Normal'],
    'Horsea': ['Water'],
    'Seadra': ['Water'],
    'Goldeen': ['Water'],
    'Seaking': ['Water'],
    'Staryu': ['Water'],
    'Starmie': ['Water', 'Psychic'],
    'MrMime': ['Psychic'], 
    'Scyther': ['Bug', 'Flying'],
    'Jynx': ['Ice', 'Psychic'],
    'Electabuzz': ['Electric'],
    'Magmar': ['Fire'],
    'Pinsir': ['Bug'],
    'Tauros': ['Normal'],
    'Magikarp': ['Water'],
    'Gyarados': ['Water', 'Flying'],
    'Lapras': ['Water', 'Ice'],
    'Ditto': ['Normal'],
    'Eevee': ['Normal'],
    'Vaporeon': ['Water'],
    'Jolteon': ['Electric'],
    'Flareon': ['Fire'],
    'Porygon': ['Normal'],
    'Omanyte': ['Rock', 'Water'],
    'Omastar': ['Rock', 'Water'],
    'Kabuto': ['Rock', 'Water'],
    'Kabutops': ['Rock', 'Water'],
    'Aerodactyl': ['Rock', 'Flying'],
    'Snorlax': ['Normal'],
    'Articuno': ['Ice', 'Flying'],
    'Zapdos': ['Electric', 'Flying'],
    'Moltres': ['Fire', 'Flying'],
    'Dratini': ['Dragon'],
    'Dragonair': ['Dragon'],
    'Dragonite': ['Dragon', 'Flying'],
    'Mewtwo': ['Psychic'],
    'Mew': ['Psychic']
}

GEN1_TYPES = [
    'Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison',
    'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon', 'Steel'
]

class PokemonDataset(Dataset):
    def __init__(self, data_dir, transform=None, images_per_type = 500):
        self.data_dir = data_dir
        self.transform = transform
        self.images_per_type = images_per_type
        self.type_to_idx = {type_name: idx for idx, type_name in enumerate(GEN1_TYPES)}
        
        # Initialize storage for balanced dataset
        self.image_paths = []
        self.labels = []
        
        # Collect and balance dataset
        self._collect_and_balance_dataset()
    
    def _collect_and_balance_dataset(self):
        # Collect and balance images by type
        print("Collecting and balancing dataset...")
        
        # First, collect all images by type
        type_to_images = defaultdict(list)
        
        for pokemon_name in os.listdir(self.data_dir):
            pokemon_dir = os.path.join(self.data_dir, pokemon_name)
            if not os.path.isdir(pokemon_dir):
                continue
            
            if pokemon_name not in GEN1_POKEMON_TYPES:
                print(f"Warning: {pokemon_name} not found in type mappings")
                continue
            
            # Get all images for this Pokemon
            pokemon_images = []
            for img_name in os.listdir(pokemon_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(pokemon_dir, img_name)
                    pokemon_images.append((img_path, pokemon_name))
            
            # Add images to each type this Pokemon belongs to
            for type_name in GEN1_POKEMON_TYPES[pokemon_name]:
                type_to_images[type_name].extend(pokemon_images)
        
        # Print initial distribution
        print("\nInitial type distribution:")
        for type_name, images in type_to_images.items():
            print(f"{type_name}: {len(images)} images")
        
        # Balance dataset by sampling images_per_type for each type
        print(f"\nBalancing dataset to {self.images_per_type} images per type...")
        
        selected_images = set()  # Track selected images to avoid duplicates
        
        for type_name in GEN1_TYPES:
            if type_name not in type_to_images:
                print(f"Warning: No images found for type {type_name}")
                continue
            
            available_images = type_to_images[type_name]
            
            if len(available_images) == 0:
                print(f"Warning: No images available for type {type_name}")
                continue
            
            # Randomly sample images for this type
            sampled_count = 0
            random.shuffle(available_images)
            
            for img_path, pokemon_name in available_images:
                if sampled_count >= self.images_per_type:
                    break
                    
                if img_path not in selected_images:
                    # Create label vector
                    label_vector = torch.zeros(len(GEN1_TYPES))
                    for pokemon_type in GEN1_POKEMON_TYPES[pokemon_name]:
                        label_vector[self.type_to_idx[pokemon_type]] = 1
                    
                    self.image_paths.append(img_path)
                    self.labels.append(label_vector)
                    selected_images.add(img_path)
                    sampled_count += 1
            
            print(f"Selected {sampled_count} images for type {type_name}")
        
        # Shuffle the dataset
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)
        
        # Print final statistics
        print("\nFinal dataset statistics:")
        type_counts = torch.sum(torch.stack(self.labels), dim=0)
        for type_name, count in zip(GEN1_TYPES, type_counts):
            print(f"{type_name}: {int(count)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class PokemonTypeClassifier(nn.Module):
    def __init__(self, num_types):
        super(PokemonTypeClassifier, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_types)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = self.sigmoid(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, patience, device='cuda'):
                
    # Train model with early stopping
    model = model.to(device)
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    # Store training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    print("\nStarting training with early stopping...")
    print(f"Maximum epochs: {num_epochs}")
    print(f"Early stopping patience: {patience}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == labels).float().sum()
            total_predictions += labels.numel()
            
            # Print batch progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).float().sum()
                val_total += labels.numel()
        
        # Calculate epoch statistics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * correct_predictions / total_predictions
        val_accuracy = 100 * val_correct / val_total
        
        # Store in history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy.item())
        history['val_acc'].append(val_accuracy.item())
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%\n')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': train_accuracy.item(),
                'val_acc': val_accuracy.item(),
                'history': history
            }, 'best_model.pth')
            print("Saved new best model!")
        else:
            epochs_without_improvement += 1
            
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs")
            print(f"Best validation loss was {best_val_loss:.4f} at epoch {best_epoch+1}")
            break
    
    return history

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up data transforms with more augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dataset with balanced type distribution
    data_dir = "./balanced-dataset"
    images_per_type = 500
    
    print("Loading and balancing dataset...")
    dataset = PokemonDataset(data_dir, transform=transform, images_per_type=images_per_type)

    # Split dataset with fixed random state
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model and training parameters
    model = PokemonTypeClassifier(num_types=len(GEN1_TYPES))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model with early stopping
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        num_epochs=50,     # Maximum number of epochs
        patience=10,       # Early stopping patience
        device=device
    )

    print("Training complete!")

if __name__ == "__main__":
    main()