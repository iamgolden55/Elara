n{Colors.GREEN}Category Summary: {cat.upper()}{Colors.ENDC}")
            print(f"  - Average Quality Score: {avg_quality:.1f}")
            print(f"  - Average Response Length: {avg_length:.1f} chars")
            print(f"  - Average Generation Time: {avg_time:.2f} seconds")
        
        # Save results if configured
        if self.config.save_results:
            results_file = os.path.join(
                self.config.results_dir, 
                f"medical_model_test_{self.config.user_type}_{timestamp}.json"
            )
            
            # Add metadata
            full_results = {
                "metadata": {
                    "timestamp": timestamp,
                    "user_type": self.config.user_type,
                    "base_model": self.config.base_model,
                    "temperature": self.config.temperature,
                    "repetition_penalty": self.config.repetition_penalty,
                    "device": str(self.device)
                },
                "results": results
            }
            
            with open(results_file, 'w') as f:
                json.dump(full_results, f, indent=2)
            
            print(f"\n{Colors.GREEN}✅ Results saved to: {results_file}{Colors.ENDC}")
        
        return results
    
    def run_interactive_mode(self):
        """Run an interactive session for testing the model"""
        print(f"\n{Colors.HEADER}🩺 Elara Medical Interactive Mode{Colors.ENDC}")
        print(f"{Colors.BOLD}User Type: {self.config.user_type}{Colors.ENDC}")
        print(f"Device: {self.device}")
        print("Type 'exit' to quit, 'settings' to change settings, or your medical query.")
        
        while True:
            print(f"\n{Colors.BLUE}Enter your query:{Colors.ENDC}")
            query = input("> ")
            
            if query.lower() == 'exit':
                print(f"{Colors.GREEN}Exiting interactive mode.{Colors.ENDC}")
                break
            
            if query.lower() == 'settings':
                self._change_settings()
                continue
            
            if not query.strip():
                continue
            
            start_time = time.time()
            response = self.generate_response(query)
            elapsed = time.time() - start_time
            
            evaluation = self.evaluate_response(response)
            
            print(f"\n{Colors.CYAN}Response:{Colors.ENDC}")
            print(f"{response}")
            print(f"\n{Colors.CYAN}Evaluation:{Colors.ENDC}")
            print(f"  - Quality Score: {evaluation['quality_score']} ({evaluation['quality_level']})")
            print(f"  - Length: {evaluation['length']} chars")
            print(f"  - Generation time: {elapsed:.2f} seconds")
    
    def _change_settings(self):
        """Change generation settings interactively"""
        print(f"\n{Colors.HEADER}Change Generation Settings:{Colors.ENDC}")
        print(f"Current settings:")
        print(f"  1. Temperature: {self.config.temperature}")
        print(f"  2. Repetition Penalty: {self.config.repetition_penalty}")
        print(f"  3. Max New Tokens: {self.config.max_new_tokens}")
        print(f"  4. User Type: {self.config.user_type}")
        print(f"  5. Return to interactive mode")
        
        choice = input("Enter number to change (1-5): ")
        
        try:
            choice = int(choice)
            if choice == 1:
                temp = input(f"Enter new temperature (0.1-1.0) [{self.config.temperature}]: ")
                if temp.strip():
                    self.config.temperature = float(temp)
            elif choice == 2:
                rep = input(f"Enter new repetition penalty (1.0-10.0) [{self.config.repetition_penalty}]: ")
                if rep.strip():
                    self.config.repetition_penalty = float(rep)
            elif choice == 3:
                tokens = input(f"Enter max new tokens (100-1000) [{self.config.max_new_tokens}]: ")
                if tokens.strip():
                    self.config.max_new_tokens = int(tokens)
            elif choice == 4:
                print("User types: medical_professional, patient, student, researcher")
                utype = input(f"Enter user type [{self.config.user_type}]: ")
                if utype.strip():
                    self.config.user_type = utype
            elif choice == 5:
                return
            
            print(f"{Colors.GREEN}Settings updated!{Colors.ENDC}")
        except:
            print(f"{Colors.RED}Invalid input. Settings unchanged.{Colors.ENDC}")

def main():
    """Main function to run the tester"""
    parser = argparse.ArgumentParser(description="Test the medical model with real-world queries")
    
    # Required arguments
    parser.add_argument("--mode", type=str, default="interactive", 
                        choices=["interactive", "all", "category", "single"],
                        help="Test mode: interactive, all categories, specific category, or single query")
    
    # Optional arguments
    parser.add_argument("--category", type=str, 
                        help="Category to test (required if mode=category)")
    parser.add_argument("--query", type=str, 
                        help="Query to test (required if mode=single)")
    parser.add_argument("--user_type", type=str, default="medical_professional",
                        choices=["medical_professional", "patient", "student", "researcher"],
                        help="User type to simulate")
    parser.add_argument("--temperature", type=float, default=0.15,
                        help="Temperature for generation")
    parser.add_argument("--rep_penalty", type=float, default=4.0,
                        help="Repetition penalty")
    parser.add_argument("--base_model", type=str, default="microsoft/DialoGPT-medium",
                        help="Base model path or identifier")
    parser.add_argument("--adapter_path", type=str, default="../models_files/medical_lora",
                        help="Path to LoRA adapter")
    
    args = parser.parse_args()
    
    # Configure test settings
    config = TestConfig(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        repetition_penalty=args.rep_penalty,
        user_type=args.user_type
    )
    
    # Initialize tester
    tester = MedicalModelTester(config)
    
    # Load model
    if not tester.load_model():
        print(f"{Colors.RED}Failed to load model. Exiting.{Colors.ENDC}")
        return
    
    # Run tests based on mode
    if args.mode == "interactive":
        tester.run_interactive_mode()
    elif args.mode == "all":
        tester.run_test()
    elif args.mode == "category":
        if not args.category:
            print(f"{Colors.RED}Error: --category is required for category mode.{Colors.ENDC}")
            return
        tester.run_test(category=args.category)
    elif args.mode == "single":
        if not args.query:
            print(f"{Colors.RED}Error: --query is required for single mode.{Colors.ENDC}")
            return
        tester.run_test(single_query=args.query)

if __name__ == "__main__":
    main()
