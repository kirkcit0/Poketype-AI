import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import random
import numpy as np
from torch import nn
import warnings
warnings.filterwarnings('ignore')

# Keep the same type configurations and model class
GEN1_TYPES = [
    'Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison',
    'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon', 'Steel'
]

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
    'Clefairy': ['Normal'],  # Note: Changed to Fairy type in Gen 6
    'Clefable': ['Normal'],  # Note: Changed to Fairy type in Gen 6
    'Vulpix': ['Fire'],
    'Ninetales': ['Fire'],
    'Jigglypuff': ['Normal'],  # Note: Changed to Normal/Fairy in Gen 6
    'Wigglytuff': ['Normal'],  # Note: Changed to Normal/Fairy in Gen 6
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
    'Magnemite': ['Electric', 'Steel'],  # Note: Added Steel type in Gen 2
    'Magneton': ['Electric', 'Steel'],   # Note: Added Steel type in Gen 2
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
    'MrMime': ['Psychic'],  # Note: Changed to Psychic/Fairy in Gen 6
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

class PokemonTester:
    def __init__(self, model_path, dataset_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.dataset_path = dataset_path
        self.prediction_threshold = 0.05  # Threshold for showing third prediction
        
        # Load model
        self.model = PokemonTypeClassifier(num_types=len(GEN1_TYPES))
        
        # Load the checkpoint dictionary
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract the model state dict from the checkpoint
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.4f}")
        else:
            # Fallback for older model format
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
        # Setup image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_path, pokemon_name):
        # Predict types for a single image and compare with actual types
        image = Image.open(image_path).convert('RGB')
        transformed_image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(transformed_image)
            probabilities = outputs[0].cpu().numpy()
        
        actual_types = GEN1_POKEMON_TYPES[pokemon_name]
        
        # Get predictions and their probabilities
        predictions = [(GEN1_TYPES[i], prob) for i, prob in enumerate(probabilities)]
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Determine how many predictions to show based on probability differences
        if len(predictions) >= 3:
            prob_diff = predictions[1][1] - predictions[2][1]
            show_third = prob_diff < self.prediction_threshold
            predictions = predictions[:3 if show_third else 2]
        else:
            predictions = predictions[:2]
        
        return predictions, actual_types, image

    def calculate_accuracy(self, num_samples=100):
        # Calculate accuracy across random samples
        correct_predictions = 0
        total_predictions = 0
        
        pokemon_folders = [f for f in os.listdir(self.dataset_path) 
                         if os.path.isdir(os.path.join(self.dataset_path, f))]
        
        for _ in range(num_samples):
            pokemon_name = random.choice(pokemon_folders)
            pokemon_dir = os.path.join(self.dataset_path, pokemon_name)
            
            image_files = [f for f in os.listdir(pokemon_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                continue
                
            image_file = random.choice(image_files)
            image_path = os.path.join(pokemon_dir, image_file)
            
            predictions, actual_types, _ = self.predict_image(image_path, pokemon_name)
            pred_types = [p[0] for p in predictions[:2]]  # Always use top 2 for accuracy
            
            # For single-type Pokémon, only consider the top prediction
            if len(actual_types) == 1:
                correct = pred_types[0] == actual_types[0]
                correct_predictions += correct
                total_predictions += 1
            # For dual-type Pokémon, consider both predictions
            else:
                correct = len(set(pred_types) & set(actual_types))
                correct_predictions += correct
                total_predictions += 2
        
        accuracy = correct_predictions / total_predictions
        print(f"\nOverall Accuracy: {accuracy:.2%}")
        return accuracy

    def show_random_predictions(self, num_images=15):
        # Display grid of random predictions
        pokemon_folders = [f for f in os.listdir(self.dataset_path) 
                         if os.path.isdir(os.path.join(self.dataset_path, f))]
        cols = 3
        rows = (num_images + cols - 1) // cols
        
        fig = plt.figure(figsize=(15, rows * 2))
        
        for idx in range(num_images):
            pokemon_name = random.choice(pokemon_folders)
            pokemon_dir = os.path.join(self.dataset_path, pokemon_name)
            
            image_files = [f for f in os.listdir(pokemon_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                continue
                
            image_file = random.choice(image_files)
            image_path = os.path.join(pokemon_dir, image_file)
            
            predictions, actual_types, image = self.predict_image(image_path, pokemon_name)
            
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(image)
            plt.axis('off')
            
            # Create concise title
            title = f"{pokemon_name}\nActual: {', '.join(actual_types)}\n"
            for type_name, prob in predictions:
                title += f"{type_name}: {prob:.1%}\n"
            
            # Enhanced color coding
            pred_types = [p[0] for p in predictions[:len(actual_types)]]
            correct_types = set(pred_types) & set(actual_types)
            
            if len(actual_types) == 1:
                # Single-type Pokémon
                color = 'green' if pred_types[0] == actual_types[0] else 'red'
            else:
                # Dual-type Pokémon
                if len(correct_types) == 2:
                    color = 'green'  # Both types correct
                elif len(correct_types) == 1:
                    color = 'orange'  # One type correct
                else:
                    color = 'red'    # No types correct
            
            plt.title(title, color=color, fontsize=8)
        
        plt.tight_layout()
        return fig

    def calculate_accuracy(self, num_samples=100):
        # Calculate accuracy across random samples with detailed statistics
        correct_predictions = 0
        partial_predictions = 0  # For dual-type Pokémon where one type is correct
        total_predictions = 0
        
        pokemon_folders = [f for f in os.listdir(self.dataset_path) 
                         if os.path.isdir(os.path.join(self.dataset_path, f))]
        
        for _ in range(num_samples):
            pokemon_name = random.choice(pokemon_folders)
            pokemon_dir = os.path.join(self.dataset_path, pokemon_name)
            
            image_files = [f for f in os.listdir(pokemon_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                continue
                
            image_file = random.choice(image_files)
            image_path = os.path.join(pokemon_dir, image_file)
            
            predictions, actual_types, _ = self.predict_image(image_path, pokemon_name)
            pred_types = [p[0] for p in predictions[:2]]  # Always use top 2 for accuracy
            
            # For single-type Pokémon
            if len(actual_types) == 1:
                correct = pred_types[0] == actual_types[0]
                correct_predictions += correct
                total_predictions += 1
            # For dual-type Pokémon
            else:
                correct = len(set(pred_types) & set(actual_types))
                if correct == 2:
                    correct_predictions += 2
                elif correct == 1:
                    partial_predictions += 1
                total_predictions += 2
        
        full_accuracy = correct_predictions / total_predictions
        partial_accuracy = partial_predictions / total_predictions
        
        print(f"\nAccuracy Statistics:")
        print(f"Full Accuracy: {full_accuracy:.2%}")
        if partial_predictions > 0:
            print(f"Partial Matches: {partial_accuracy:.2%}")
        print(f"Overall Accuracy: {(full_accuracy + partial_accuracy/2):.2%}")
        
        return full_accuracy

def main():
    # Paths
    model_path = "best_model.pth"
    dataset_path = "./balanced-dataset"
    
    # Create tester
    tester = PokemonTester(model_path, dataset_path)
    
    # Calculate and print accuracy
    accuracy = tester.calculate_accuracy(num_samples=100)
    
    # Show predictions
    fig = tester.show_random_predictions(num_images=15)
    plt.show()

if __name__ == "__main__":
    main()