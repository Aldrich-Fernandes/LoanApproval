for x, letter in enumerate(s):
            matched = False
            if len(s) == 2 and s in ValidCases:
                return True
            
            if s[x:x+2] in ValidCases:
                s.remove(s[x:x+2])
            else:
                for y in range(x, len(s)-1, 3):
                    if letter + s[y] in ValidCases:
                        matched = True
            
                if not matched:
                    return False